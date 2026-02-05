#ifndef TINYTENSOR
#define TINYTENSOR

#include <memory>
#include <iostream>

#include "cuda_bridge.h"
#include "toolbox.h"
#include "kernels.h"

enum class DeviceStatus {HostOnly, DeviceOnly, Synced};

template <typename T>
class TinyTensor {
	private:
		// TinyTensor 长度,行列
		size_t size{};
		size_t row{};
		size_t col{};
		
		// TinyTensor 内存管理
		std::unique_ptr <T[]> data{nullptr};  	// CPU
		T * data_gpu{nullptr};				  	// GPU

		// TinyTensor Grad 梯度 懒加载
		bool flag_grad{false};
		std::shared_ptr<TinyTensor<T>> grad{nullptr};		
		
		// TinyTensor 状态管理
		DeviceStatus status{DeviceStatus::HostOnly};

	public:
		// 构造函数
		TinyTensor(size_t row=0, size_t col=0) : row{row}, col{col} {
			if(row > 0 && col > 0) {
				size = row * col;
				data.reset(new T[size]{});
			}
        }

		// 复制构造函数, 复制赋值函数（禁用）
		TinyTensor(const TinyTensor& tensor1) = delete;
		TinyTensor& operator=(const TinyTensor& tensor1) = delete;

		// 移动构造函数
		TinyTensor(TinyTensor&& tensor1) noexcept
			: 	size{tensor1.size}, row{tensor1.row}, col{tensor1.col},
			 	data{std::move(tensor1.data)}, data_gpu{tensor1.data_gpu},
				grad{std::move(tensor1.grad)},
				flag_grad{tensor1.flag_grad}, status{tensor1.status} {
			tensor1.size = 0;
			tensor1.row = 0;
			tensor1.col = 0;
			tensor1.data_gpu = nullptr;
			tensor1.flag_grad = false;
			tensor1.status = DeviceStatus::HostOnly;
		}

		// 移动赋值函数
		TinyTensor& operator=(TinyTensor&& tensor1) noexcept {
			// 自我赋值检测
			if (&tensor1 == this) return *this;

			// 释放自我
			if(data_gpu) gpu_free(data_gpu);

			// 移动赋值
			size = tensor1.size;
			row = tensor1.row;
			col = tensor1.col;
			data = std::move(tensor1.data);
			grad = std::move(tensor1.grad);
			data_gpu = tensor1.data_gpu;
			flag_grad = tensor1.flag_grad;
			status = tensor1.status;

			//源对象置空
			tensor1.size = 0;
    		tensor1.row = 0;
    		tensor1.col = 0;
    		tensor1.data_gpu = nullptr;

			return *this;
		}

		// 析构函数
		~TinyTensor() {
    		if(data_gpu) gpu_free(data_gpu);
		}
		
		// 获取长度
		size_t get_size() { return size; }
		// 获取data指针
		T* get_ptr() { return data.get(); }
		// 获取dataCUDA指针
		T* get_cuda_ptr() { return data_gpu; }
		// 获取梯度指针
		TinyTensor<T>& get_grad() {
			if (!flag_grad || !grad) {
				throw std::runtime_error("该 Tensor 未开启梯度计算功能！");
			}
			return *grad;
   		}

		// 获取TinyTensor状态
		DeviceStatus get_status() { return status; }

		// 一维下标访问
		T& operator[](size_t index) const { return data[index]; }

		// 二维下标访问
		T& operator()(size_t row_, size_t col_) const { return data[row_ * col + col_]; }
		
		// 算子 + 重载
		TinyTensor<T> operator+(const TinyTensor<T> & other) const {
			if (row != other.row || col != other.col) {
				throw std::runtime_error("维度不一致");
			}
			
			TinyTensor<T> result(row, col);

			if(status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				result.allocate_device();
				launch_matrix_add(data_gpu, other.data_gpu, result.data_gpu, row, col);
				result.status = DeviceStatus::DeviceOnly;	
			} else {
				add(data.get(), other.data.get(), result.data.get(), size);
			}

			return result;
		}
		
		// 算子 * 重载
		TinyTensor<T> operator*(const TinyTensor<T> & other) const {
			if (col != other.row) {
				throw std::runtime_error("维度不一致");
			}
			
			TinyTensor<T> result(row, other.col);

			if(status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				result.allocate_device();
				launch_matrix_mul_tiled(data_gpu, other.data_gpu, result.data_gpu, row, col, other.col);
				result.status = DeviceStatus::DeviceOnly;
			} else {
				matmul(data.get(), other.data.get(), result.data.get(), row, col, other.col);
			}
			return result;
		}

		// 算子 * 重载-数乘 
		TinyTensor<T> operator*(float alpha) {
			TinyTensor<T> result(row, col);
			if(status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				result.allocate_device();
				launch_mul_num(data_gpu, result.data_gpu, alpha, size);
				result.status = DeviceStatus::DeviceOnly;
			} else {
				func_scalar(data.get(), result.data.get(), alpha, size);
        	}
			return result;
		}

		// 算子 *= 重载
		TinyTensor<T>& operator*=(float alpha) {
			if(status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				launch_scale(data_gpu, alpha, size);
			} else {
				scale(data.get(), alpha, size);
        	}
			return *this;
		}

		// 填充函数
		void fill(T value) {
			T* begin = data.get();
			for (size_t i = 0; i < size; ++i) {
            	begin[i] = value;
        	}
			status = DeviceStatus::HostOnly;
        }

		// 打印函数
		void print() {
			std::cout << "size:" << size << '\n';
			std::cout << "Data:";
			T* begin = data.get();
			 for (size_t i = 0; i < size; ++i) {
            	std::cout << begin[i] << ' ';
        	}
			std::cout << '\n';		
        }

		// Device 状态
		TinyTensor& to(const std::string& device) {
			if(device =="cuda" || device =="gpu") {
				sync_to_device();
			} else if(device =="cpu") {
				sync_to_host();
			} else {
				std::cerr << "Unknow device:" << device << '\n';
			}
			return *this;
		}

		// Grad 状态
		void set_grad(bool state_grad) {
			this->flag_grad = state_grad;
			if (state_grad && !grad) {
				grad = std::make_shared<TinyTensor<T>>(row, col);
				grad->flag_grad = false; 
				grad->grad = nullptr;
    		}
		}

		// CUDA allocate
		void allocate_device() {
			if (data_gpu) gpu_free(data_gpu);
    		data_gpu = static_cast<T*>(gpu_malloc(size * sizeof(T)));
		}
		
		// 复制数据到GPU
		void copy_to_device() {
			gpu_memcpy_h2d(data_gpu, data.get(), size * sizeof(T));
		}
		
		// 复制数据到CPU
		void copy_to_host() {
			gpu_memcpy_d2h(data.get(), data_gpu, size * sizeof(T));
		}

		// 复制数据到GPU
		void sync_to_device() {
			if(status == DeviceStatus::HostOnly) {
				if(!data_gpu) allocate_device();
				copy_to_device();
				status = DeviceStatus::Synced;
			}
			if(flag_grad && grad) {
				grad->to("gpu");
			}
		}
		
		// 复制数据到CPU
		void sync_to_host() {
			if (status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				copy_to_host();
				status = DeviceStatus::Synced;
			}
			if(flag_grad && grad) {
				grad->to("cpu");
			}
		}

		// 激活函数
		void relu() {
			if(data_gpu) {
				launch_relu(data_gpu, size);
			} else {
				for (size_t i = 0; i < size; ++i) {
                	data[i] = data[i] > 0 ? data[i] : 0;
            	}
			}
		};
		
		// 转置
		TinyTensor<T> trans() const {
			TinyTensor<T> result{col, row};
			if(status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				result.allocate_device();
				launch_transpose(data_gpu, result.data_gpu, row, col);
				result.status = DeviceStatus::DeviceOnly;
			} else {
				func_transpose(data.get(), result.get_ptr(), row, col);
				result.status = DeviceStatus::HostOnly;
        	}
			return result;
		}

		// 求和
		TinyTensor<T> sum(bool axis) {
			size_t out_row = (axis == 0) ? 1 : row;
			size_t out_col = (axis == 0) ? col : 1;
			TinyTensor<T> result(out_row, out_col);

			if (status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				result.allocate_device();
				launch_sum(data_gpu, result.data_gpu, row, col, axis);
				result.status = DeviceStatus::DeviceOnly;
			} else {
				// CPU 路径
				if (axis == 0) {
					for (size_t c = 0; c < col; ++c) {
						T s = 0;
						for (size_t r = 0; r < row; ++r) {
							s += (*this)(r, c);
						}
						result(0, c) = s;
					}
				} else {
					for (size_t r = 0; r < row; ++r) {
						T s = 0;
						for (size_t c = 0; c < col; ++c) {
							s += (*this)(r, c);
						}
						result(r, 0) = s;
					}
				}
				result.status = DeviceStatus::HostOnly;
			}
			return result;
		}

		// 梯度置零
		void grad_zero() {
			if(flag_grad && grad) {
       			grad->fill(0);
        		if(grad->data_gpu) gpu_memset_zero(grad->data_gpu, size * sizeof(T));
    		}
		}	

};

#endif
