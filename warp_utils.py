import torch
import ImageWarp

class image_warp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image:torch.Tensor, motion_vector:torch.Tensor):
        ctx.save_for_backward(image, motion_vector)
        output = ImageWarp.warp_forward(image, motion_vector)
        return output

    @staticmethod
    def backward(ctx, backprop):
        image_grad_output = ImageWarp.backward(*ctx.saved_tensors, backprop)
        return image_grad_output, None
        
class image_warp_accum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_src:torch.Tensor, image_dst:torch.Tensor, motion_vector:torch.Tensor, alpha=0.5):
        ctx.save_for_backward(image_src, image_dst, motion_vector)
        ctx.alpha = alpha
        output = ImageWarp.warp_accum_forward(image_src, image_dst, motion_vector, alpha)
        return output

    @staticmethod
    def backward(ctx, backprop):
        img_src_grad_output, img_dst_grad_output = ImageWarp.warp_accum_backward(*ctx.saved_tensors, ctx.alpha, backprop)
        return img_src_grad_output, img_dst_grad_output, None, None

