

__global__ void ImageWarpKernel(
	int pixel_count,
	const float* image, 
	const float* motion_vector, 
	const int batch,
	const int height,
	const int width,
	const int channel,
	float* output
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = i % width;
	const int h = (i / width) % height;
	const int b = (i / (width * height)) % batch;

	// const int sx = 1;
	const int sy = width;
	const int sc = width * height;
	const int sb = channel * width * height;

	const int motion_vector_base = b*sc*2+h*sy*2+w*2;

	const float dx = motion_vector[motion_vector_base];
	const float dy = motion_vector[motion_vector_base+1];

	// (yy, xx) is the reprojected position
	const float yy = dy + h ;
	const float xx = dx + w;
	
	const int yy_int = round(yy);
	const int xx_int = round(xx);
	const float dyy = yy - yy_int;
	const float dxx = xx - xx_int;

	// xl, xr, yu, yb are the coordinates of the neibours near the reprojected position
	int xl = 0;
	int xr = 0;
	int yu = 0;
	int yb = 0;

	if (yy_int >= 0 && yy_int < height && xx_int >= 0 && xx_int < width)
	{
		if (dyy > 0) {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int;
				yb = yy_int+1;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int;
				yb = yy_int+1;
			}
		}
		else {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int-1;
				yb = yy_int;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int-1;
				yb = yy_int;
			}
		}
		
		const float u_diff    = abs(yy-yu);
		const float b_diff    = abs(yy-yb);
		const float l_diff    = abs(xx-xl);
		const float r_diff    = abs(xx-xr);
		const float ul_weight = 1-u_diff*l_diff;
		const float ur_weight = 1-u_diff*r_diff;
		const float bl_weight = 1-b_diff*l_diff;
		const float br_weight = 1-b_diff*r_diff;
		const float total     = ul_weight+ur_weight+bl_weight+br_weight;

		const int batch_part = b*sb;
		const int this_idx_base = batch_part+h*sy+w;
		
		if (yu >= 0 && yu < height && xl >= 0 && xl < width) {
			const float weight = ul_weight / total;
			const int   idx_base = batch_part + yu*sy + xl;
			for (int c = 0; c < channel; c++) {
				output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
			}
		}
		if (yu >= 0 && yu < height && xr >= 0 && xr < width) {
			const float weight = ur_weight / total;
			const int   idx_base = batch_part + yu*sy + xr;
			for (int c = 0; c < channel; c++) {
				output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xl >= 0 && xl < width) {
			const float weight = bl_weight / total;
			const int   idx_base  = batch_part + yb*sy + xl;
			for (int c = 0; c < channel; c++) {
				output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xr >= 0 && xr < width) {
			const float weight = br_weight / total;
			const int   idx_base  = batch_part + yb*sy + xr;
			for (int c = 0; c < channel; c++) {
				output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
			}
		}
	}
	
}

__global__ void ImageWarpAccumKernel(
	int pixel_count,
	const float* image, 
	const float* motion_vector, 
	const int batch,
	const int height,
	const int width,
	const int channel,
	const float alpha,
	float* output
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = i % width;
	const int h = (i / width) % height;
	const int b = (i / (width * height)) % batch;

	// const int sx = 1;
	const int sy = width;
	const int sc = width * height;
	const int sb = channel * width * height;

	const int motion_vector_base = b*sc*2+h*sy*2+w*2;

	const float dx = motion_vector[motion_vector_base];
	const float dy = motion_vector[motion_vector_base+1];

	// (yy, xx) is the reprojected position
	const float yy = dy + h ;
	const float xx = dx + w;
	
	const int yy_int = round(yy);
	const int xx_int = round(xx);
	const float dyy = yy - yy_int;
	const float dxx = xx - xx_int;

	// xl, xr, yu, yb are the coordinates of the neibours near the reprojected position
	int xl = 0;
	int xr = 0;
	int yu = 0;
	int yb = 0;

	if (yy_int >= 0 && yy_int < height && xx_int >= 0 && xx_int < width)
	{
		if (dyy > 0) {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int;
				yb = yy_int+1;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int;
				yb = yy_int+1;
			}
		}
		else {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int-1;
				yb = yy_int;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int-1;
				yb = yy_int;
			}
		}
		
		const float u_diff    = abs(yy-yu);
		const float b_diff    = abs(yy-yb);
		const float l_diff    = abs(xx-xl);
		const float r_diff    = abs(xx-xr);
		const float ul_weight = 1-u_diff*l_diff;
		const float ur_weight = 1-u_diff*r_diff;
		const float bl_weight = 1-b_diff*l_diff;
		const float br_weight = 1-b_diff*r_diff;
		const float total     = ul_weight+ur_weight+bl_weight+br_weight;

		const int batch_part = b*sb;
		const int this_idx_base = batch_part+h*sy+w;

		float color[64] = {0};
		if (yu >= 0 && yu < height && xl >= 0 && xl < width) {
			const float weight = ul_weight / total;
			const int   idx_base = batch_part + yu*sy + xl;
			for (int c = 0; c < channel; c++) {
				color[c] += image[idx_base+c*sc] * weight;
			}
		}
		if (yu >= 0 && yu < height && xr >= 0 && xr < width) {
			const float weight = ur_weight / total;
			const int   idx_base = batch_part + yu*sy + xr;
			for (int c = 0; c < channel; c++) {
				color[c] += image[idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xl >= 0 && xl < width) {
			const float weight = bl_weight / total;
			const int   idx_base  = batch_part + yb*sy + xl;
			for (int c = 0; c < channel; c++) {
				color[c] += image[idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xr >= 0 && xr < width) {
			const float weight = br_weight / total;
			const int   idx_base  = batch_part + yb*sy + xr;
			for (int c = 0; c < channel; c++) {
				color[c] += image[idx_base+c*sc] * weight;
			}
		}
		for (int c = 0; c < channel; c++) {
			output[this_idx_base+c*sc] = output[this_idx_base+c*sc] * (1-alpha) + alpha * color[c];
		}
	}
}

__global__ void ImageWarpGradKernel(
	int pixel_count,
	const float* image, 
	const float* motion_vector, 
	const float* backprop, 
	const int batch,
	const int height,
	const int width,
	const int channel,
	float* output
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = i % width;
	const int h = (i / width) % height;
	const int b = (i / (width * height)) % batch;

	// const int sx = 1;
	const int sy = width;
	const int sc = width * height;
	const int sb = channel * width * height;

	const int motion_vector_base = b*sc*2+h*sy*2+w*2;

	const float dx = motion_vector[motion_vector_base];
	const float dy = motion_vector[motion_vector_base+1];

	// (yy, xx) is the reprojected position
	const float yy = dy + h ;
	const float xx = dx + w;
	
	const int yy_int = round(yy);
	const int xx_int = round(xx);
	const float dyy = yy - yy_int;
	const float dxx = xx - xx_int;

	// xl, xr, yu, yb are the coordinates of the neibours near the reprojected position
	int xl = 0;
	int xr = 0;
	int yu = 0;
	int yb = 0;

	if (yy_int >= 0 && yy_int < height && xx_int >= 0 && xx_int < width)
	{
		if (dyy > 0) {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int;
				yb = yy_int+1;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int;
				yb = yy_int+1;
			}
		}
		else {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int-1;
				yb = yy_int;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int-1;
				yb = yy_int;
			}
		}
		
		const float u_diff    = abs(yy-yu);
		const float b_diff    = abs(yy-yb);
		const float l_diff    = abs(xx-xl);
		const float r_diff    = abs(xx-xr);
		const float ul_weight = 1-u_diff*l_diff;
		const float ur_weight = 1-u_diff*r_diff;
		const float bl_weight = 1-b_diff*l_diff;
		const float br_weight = 1-b_diff*r_diff;
		const float total     = ul_weight+ur_weight+bl_weight+br_weight;

		const int batch_part = b*sb;
		const int this_idx_base = batch_part+h*sy+w;
		
		if (yu >= 0 && yu < height && xl >= 0 && xl < width) {
			const float weight = ul_weight / total;
			const int   idx_base = batch_part + yu*sy + xl;
			for (int c = 0; c < channel; c++) {
				// output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
				output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight;
			}
		}
		if (yu >= 0 && yu < height && xr >= 0 && xr < width) {
			const float weight = ur_weight / total;
			const int   idx_base = batch_part + yu*sy + xr;
			for (int c = 0; c < channel; c++) {
				// output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
				output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xl >= 0 && xl < width) {
			const float weight = bl_weight / total;
			const int   idx_base  = batch_part + yb*sy + xl;
			for (int c = 0; c < channel; c++) {
				// output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
				output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight;
			}
		}
		if (yb >= 0 && yb < height && xr >= 0 && xr < width) {
			const float weight = br_weight / total;
			const int   idx_base  = batch_part + yb*sy + xr;
			for (int c = 0; c < channel; c++) {
				// output[this_idx_base+c*sc] += image[idx_base+c*sc] * weight;
				output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight;
			}
		}
	}	
}

__global__ void ImageWarpAccumGradKernel(
	int pixel_count,
	const float* image, 
	const float* motion_vector, 
	const float* backprop, 
	const int batch,
	const int height,
	const int width,
	const int channel,
	const float alpha,
	float* img_grad_output,
	float* imgdst_grad_output
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = i % width;
	const int h = (i / width) % height;
	const int b = (i / (width * height)) % batch;

	// const int sx = 1;
	const int sy = width;
	const int sc = width * height;
	const int sb = channel * width * height;

	const int motion_vector_base = b*sc*2+h*sy*2+w*2;

	const float dx = motion_vector[motion_vector_base];
	const float dy = motion_vector[motion_vector_base+1];

	// (yy, xx) is the reprojected position
	const float yy = dy + h ;
	const float xx = dx + w;
	
	const int yy_int = round(yy);
	const int xx_int = round(xx);
	const float dyy = yy - yy_int;
	const float dxx = xx - xx_int;

	// xl, xr, yu, yb are the coordinates of the neibours near the reprojected position
	int xl = 0;
	int xr = 0;
	int yu = 0;
	int yb = 0;

	const int batch_part = b*sb;
	const int this_idx_base = batch_part+h*sy+w;

	if (yy_int >= 0 && yy_int < height && xx_int >= 0 && xx_int < width)
	{
		if (dyy > 0) {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int;
				yb = yy_int+1;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int;
				yb = yy_int+1;
			}
		}
		else {
			if (dxx > 0) {
				xl = xx_int;
				xr = xx_int+1;
				yu = yy_int-1;
				yb = yy_int;
			}
			else {
				xl = xx_int-1;
				xr = xx_int;
				yu = yy_int-1;
				yb = yy_int;
			}
		}
		
		const float u_diff    = abs(yy-yu);
		const float b_diff    = abs(yy-yb);
		const float l_diff    = abs(xx-xl);
		const float r_diff    = abs(xx-xr);
		const float ul_weight = 1-u_diff*l_diff;
		const float ur_weight = 1-u_diff*r_diff;
		const float bl_weight = 1-b_diff*l_diff;
		const float br_weight = 1-b_diff*r_diff;
		const float total     = ul_weight+ur_weight+bl_weight+br_weight;

		if (yu >= 0 && yu < height && xl >= 0 && xl < width) {
			const float weight = ul_weight / total;
			const int   idx_base = batch_part + yu*sy + xl;
			for (int c = 0; c < channel; c++) {
				// color[c] += image[idx_base+c*sc] * weight;
				img_grad_output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight * alpha;
			}
		}
		if (yu >= 0 && yu < height && xr >= 0 && xr < width) {
			const float weight = ur_weight / total;
			const int   idx_base = batch_part + yu*sy + xr;
			for (int c = 0; c < channel; c++) {
				// color[c] += image[idx_base+c*sc] * weight;
				img_grad_output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight * alpha;
			}
		}
		if (yb >= 0 && yb < height && xl >= 0 && xl < width) {
			const float weight = bl_weight / total;
			const int   idx_base  = batch_part + yb*sy + xl;
			for (int c = 0; c < channel; c++) {
				// color[c] += image[idx_base+c*sc] * weight;
				img_grad_output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight * alpha;
			}
		}
		if (yb >= 0 && yb < height && xr >= 0 && xr < width) {
			const float weight = br_weight / total;
			const int   idx_base  = batch_part + yb*sy + xr;
			for (int c = 0; c < channel; c++) {
				// color[c] += image[idx_base+c*sc] * weight;
				img_grad_output[idx_base+c*sc] += backprop[this_idx_base+c*sc] * weight * alpha;
			}
		}
		for (int c = 0; c < channel; c++) {
			imgdst_grad_output[this_idx_base+c*sc] = backprop[this_idx_base+c*sc] * (1-alpha);
		}
	}
	else {
		for (int c = 0; c < channel; c++) {
			imgdst_grad_output[this_idx_base+c*sc] = backprop[this_idx_base+c*sc];
		}
	}
}


void ImageWarpKernelLauncher(
	const float* image,
	const float* motion_vector,
	const int* image_size,
	float* output
) {
	int batch = image_size[0];
	int channel = image_size[1];
	int height = image_size[2];
	int width = image_size[3];

	int pixel_count = batch * height * width;
	if (pixel_count > 0) {
		dim3 GRID_SIZE((pixel_count + 1023) / 1024);
		dim3 BLOCK_SIZE(1024);
		ImageWarpKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			pixel_count,
			image,
			motion_vector,
			batch,
			height,
			width,
			channel,
			output);
	}
}

void ImageWarpAccumKernelLauncher(
	const float* image,
	const float* motion_vector,
	const int* image_size,
	const float alpha,
	float* output
) {
	int batch = image_size[0];
	int channel = image_size[1];
	int height = image_size[2];
	int width = image_size[3];

	int pixel_count = batch * height * width;
	if (pixel_count > 0) {
		dim3 GRID_SIZE((pixel_count + 1023) / 1024);
		dim3 BLOCK_SIZE(1024);
		ImageWarpAccumKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			pixel_count,
			image,
			motion_vector,
			batch,
			height,
			width,
			channel,
			alpha,
			output);
	}
}
void ImageWarpGradKernelLauncher(
	const float* image,
	const float* motion_vector,
	const float* backprop,
	const int* image_size,
	float* output
) {
	int batch = image_size[0];
	int channel = image_size[1];
	int height = image_size[2];
	int width = image_size[3];

	int pixel_count = batch * height * width;
	if (pixel_count > 0) {
		dim3 GRID_SIZE((pixel_count + 1023) / 1024);
		dim3 BLOCK_SIZE(1024);
		ImageWarpGradKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			pixel_count,
			image,
			motion_vector,
			backprop,
			batch,
			height,
			width,
			channel,
			output);
	}
}

void ImageWarpAccumGradKernelLauncher(
	const float* image,
	const float* motion_vector,
	const float* backprop,
	const int* image_size,
	const float alpha,
	float* img_grad_output,
	float* imgdst_grad_output
) {
	int batch = image_size[0];
	int channel = image_size[1];
	int height = image_size[2];
	int width = image_size[3];

	int pixel_count = batch * height * width;
	if (pixel_count > 0) {
		dim3 GRID_SIZE((pixel_count + 1023) / 1024);
		dim3 BLOCK_SIZE(1024);
		ImageWarpAccumGradKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
			pixel_count,
			image,
			motion_vector,
			backprop,
			batch,
			height,
			width,
			channel,
			alpha,
			img_grad_output,
			imgdst_grad_output);
	}
}