#ifndef TINYTENSOR
#define TINYTENSOR

#include <memory>
#include <iostream>

template <typename T>
class TinyTensor {
	private:
		// TinyTensor 内存管理
		std::unique_ptr <T[]> data{nullptr};
		// TinyTensor 长度
		size_t size{};
		// TinyTensor 行列
		size_t row{};
		size_t col{};

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
			  data{std::move(tensor1.data)} {
			tensor1.size = 0;
			tensor1.row = 0;
			tensor1.col = 0;
		}

		// 移动赋值函数
		TinyTensor& operator=(TinyTensor&& tensor1) noexcept {
			// 自我赋值检测
			if (& tensor1 == this)
				return *this;

			// 移动赋值
			size = tensor1.size;
			row = tensor1.row;
			col = tensor1.col;
			data = std::move(tensor1.data);

			return *this;
		}

		// 析构函数
		~TinyTensor() = default;
		
		// 填充函数
		void fill(T value) {
			T* begin = data.get();
			for (size_t i = 0; i < size; ++i) {
            	begin[i] = value;
        	}		
        }	

		// 下标访问
		T& operator[](int subscript) {
			return data[subscript];
        }

		// 二维下标访问
		T& operator()(int row_, int col_) {
			return data[row_ * col + col_];
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
