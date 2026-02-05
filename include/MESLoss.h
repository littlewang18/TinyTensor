#ifndef MESLOSS
#define MESLOSS

#include "TinyTensor"

template <typename T>
class MSELoss {
    public:
        T forward(TinyTensor<T>& pred, TinyTensor<T>& target) {
            if(pred.get_status() == DeviceStatus::HostOnly) {
                return mse_forward(pred.get_grad_ptr(), target.get_grad_ptr(), pred.get_size());
            } else {
                return launch_mse_forward(pred.get_grad_gpu_ptr(), target.get_grad_gpu_ptr(), pred.get_size())
            }
        }

        void backward(TinyTensor<T>& pred, TinyTensor<T>& target) {
            if(pred.get_status() == DeviceStatus::HostOnly) {
                mse_backward(pred.get_grad_ptr(), target.get_grad_ptr(), pred.get_grad_ptr(), pred.get_size());
            } else {
                launch_mse_backward(pred.get_grad_gpu_ptr(),, target.get_grad_gpu_ptr(),pred.get_grad_gpu_ptr(), pred.get_size())
            }
        }

}

#endif