#ifndef TINYTENSOR
#define TINYTENSOR

#include <memory>

template <typename T>
class TinyTensor
{
	private:
		// TinyTensor 长度
		int length{};

		// TinyTensor 头指针
		std::unique_ptr <T> hpoint{nullptr};

		// TinyTensor 尾指针
		std::unique_ptr <T> tpoint{nullptr};

	public:
		// 构造函数
		TinyTensor(int length);

		// 复制构造函数（禁用）
		TinyTensor(const TinyTensor& tensor1) = delete;
		
		// 复制赋值函数（禁用）
		TinyTensor& operator=(const TinyTensor& tensor1) = delete

		// 移动构造函数
		TinyTensor(const TinyTensor&& tensor1) noexcept;

		// 移动赋值函数
		TinyTensor& operator==(const TinyTensor&& tensor1) noexcept;

		// 析构函数
		~TinyTensor();
		
		// 填充函数
		void fill(T value);

		// 下标访问
		T operator[](int subscript);

		// 打印函数
		void print();

		// 元数据获取底层指针
		std::unique_ptr <T>* data();
		
		// 元数据获取长度
		int size();
}

#endif
