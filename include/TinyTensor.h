#ifndef TINYTENSOR
#define TINYTENSOR

#include <memory>
#include <iostream>

#include "cuda_bridge.h"


template <typename T>
class TinyTensor {
	private:
		
		// TinyTensor 长度,行列
		size_t size{};
		size_t row{};
		size_t col{};
		
		// TinyTensor 内存管理
		std::unique_ptr <T[]> data{nullptr};
		
		// CUDA 适配
		T * data_device{nullptr};

	public:
		// 构造函数
		TinyTensor(size_t row=0, size_t col=0) 
			: row{row}, col{col} {
			if(row > 0 && col > 0) {
				size = row * col;
				data.reset(new T[size]{});
			}
        }

		// 复制构造函数（禁用）
		TinyTensor(const TinyTensor& tensor1) = delete;
		
		// 复制赋值函数（禁用）
		TinyTensor& operator=(const TinyTensor& tensor1) = delete;

		// 移动构造函数
		TinyTensor(TinyTensor&& tensor1) noexcept
			: size{tensor1.size}, 
			  row{tensor1.row},
			  col{tensor1.col},
			  data{std::move(tensor1.data)},
			  data_device{tensor1.data_device} {
			tensor1.size = 0;
			tensor1.row = 0;
			tensor1.col = 0;
			tensor1.data_device = nullptr;
		}

		// 移动赋值函数
		TinyTensor& operator=(TinyTensor&& tensor1) noexcept {
			// 自我赋值检测
			if (& tensor1 == this)
				return *this;

			// 释放自我
			if(data_device) cudaFree(data_device);

			// 移动赋值
			size = tensor1.size;
			row = tensor1.row;
			col = tensor1.col;
			data = std::move(tensor1.data);
			data_device = tensor1.data_device;

			//源对象置空
			tensor1.size = 0;
    		tensor1.row = 0;
    		tensor1.col = 0;
    		tensor1.data_device = nullptr;

			return *this;
		}

		// 析构函数
		~TinyTensor();
		
		// 下标访问
		T& operator[](int subscript) {
			return data[subscript];
        }

		// 二维下标访问
		T& operator()(int row_, int col_) {
			return data[row_ * col + col_];
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

		// 元数据获取底层指针
		T* get_ptr() {
			return data.get();
        }

		// 获取底层CUDA指针
		T* get_cuda_ptr() {
			return data_device;
		}
		
		// 元数据获取长度
		size_t size_data() {
            return size;
        }

		// CUDA 适配
		void allocate_device();

		void copy_to_device();

		void copy_to_host();
};


template <typename T>
TinyTensor<T>::~TinyTensor() {
    data.reset();
    if (data_device) {
        gpu_free(data_device);
    }
}

template <typename T>
void TinyTensor<T>::allocate_device() {
    if (data_device) gpu_free(data_device);
    data_device = static_cast<T*>(gpu_malloc(size * sizeof(T)));
}

template <typename T>
void TinyTensor<T>::copy_to_device() {
    gpu_memcpy_h2d(data_device, data.get(), size * sizeof(T));
}

template <typename T>
void TinyTensor<T>::copy_to_host() {
    gpu_memcpy_d2h(data.get(), data_device, size * sizeof(T));
};

#endif
