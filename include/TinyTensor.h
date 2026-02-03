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
		std::unique_ptr <T[]> data{nullptr};  	//CPU
		T * data_gpu{nullptr};				  	//GPU

		// TinyTensor Grad 梯度
		std::unique_ptr <T[]> grad{nullptr}; 
		
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
				status{tensor1.status} {
			tensor1.size = 0;
			tensor1.row = 0;
			tensor1.col = 0;
			tensor1.data_gpu = nullptr;
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
			data_gpu = tensor1.data_gpu;
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
    		if (data_gpu) gpu_free(data_gpu);
		}
		
		// 获取长度
		size_t get_size() { return size; }
		// 获取底层指针
		T* get_ptr() { return data.get(); }
		// 获取底层CUDA指针
		T* get_cuda_ptr() { return data_gpu; }

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
				launch_matrix_add(data_gpu, other.data_gpu, result.data_gpu,
					 row, col);
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
				launch_matrix_mul_tiled(data_gpu, other.data_gpu, result.data_gpu, 
									row, col, other.col);
				result.status = DeviceStatus::DeviceOnly;
			} else {
				matmul(data.get(), other.data.get(), result.data.get(), row, col, other.col);
			}
			return result;
		}

		//算子 *= 重载
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
		}
		
		// 复制数据到CPU
		void sync_to_host() {
			if (status == DeviceStatus::DeviceOnly || status == DeviceStatus::Synced) {
				copy_to_host();
				status = DeviceStatus::Synced;
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
		}
		


};

#endif
