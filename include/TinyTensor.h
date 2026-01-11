#ifndef TINYTENSOR
#define TINYTENSOR

#include <memory>
#include <iostream>

template <typename T>
class TinyTensor {
	private:
		// TinyTensor 长度
		size_t size{};
		// TinyTensor 内存管理
		std::unique_ptr <T[]> data{nullptr};

	public:
		// 构造函数
		TinyTensor(size_t size=0) : size{size} {
			if(size > 0) {
				data.reset(new T[size]{});
			}
        }

		// 复制构造函数（禁用）
		TinyTensor(const TinyTensor& tensor1) = delete;
		
		// 复制赋值函数（禁用）
		TinyTensor& operator=(const TinyTensor& tensor1) = delete;

		// 移动构造函数
		TinyTensor(TinyTensor&& tensor1) noexcept
			: size{tensor1.size}, data{std::move(tensor1.data)} {
			tensor1.size = 0;
			tensor1.data.release();
		}

		// 移动赋值函数
		TinyTensor& operator=(TinyTensor&& tensor1) noexcept {
			// 自我赋值检测
			if (& tensor1 == this)
				return *this;

			// 释放自我
			size = 0;
			data.reset();

			// 移动赋值
			size = tensor1.size;
			data = tensor1.data;

			tensor1.size = 0;
			tensor1.data.release();

			return *this;
		}

		// 析构函数
		~TinyTensor() {
			size = 0;
			data.reset();
        }	
		
		// 填充函数
		void fill(T value) {
			T* begin = data.get();
			for (size_t i = 0; i < size; ++i) {
            	begin[i] = value;
        	}		
        }	

		// 下标访问
		T operator[](int subscript) {
			return *(data + subscript);
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
		T* data_() {
			return data.get();
        }
		
		// 元数据获取长度
		size_t size_data() {
            return size;
        }
};

#endif
