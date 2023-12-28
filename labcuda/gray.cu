#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define BLUR_SIZE 3
#define CHANNELS 3
#define uc unsigned char

using namespace cv;

__global__
void colorToGreyscaleConversion(uc* Pout, unsigned
	char* Pin, int width, int height) {
	
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	if (Col < width && Row < height) {
		// get 1D coordinate for the grayscale image
		int greyOffset = Row * width + Col;
		// one can think of the RGB image having
		// CHANNEL times columns than the grayscale image
		int rgbOffset = greyOffset * CHANNELS;
		uc r = Pin[rgbOffset]; // red value for pixel
		uc g = Pin[rgbOffset + 1]; // green value for pixel
		uc b = Pin[rgbOffset + 2]; // blue value for pixel
		// perform the rescaling and store it
		// We multiply by floating point constants
		Pout[rgbOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
		Pout[rgbOffset+1] = 0.21f * r + 0.71f * g + 0.07f * b;
		Pout[rgbOffset+2] = 0.21f * r + 0.71f * g + 0.07f * b;
		

	}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <nombre_de_archivo_imagen>" << std::endl;
        return -1;
    }

    int width, height;
    Mat image1 = imread(argv[1], IMREAD_COLOR);

    if (image1.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    namedWindow("Imagen Original", WINDOW_AUTOSIZE);
    imshow("Imagen Original", image1);
    waitKey(0);

    Size imageSize = image1.size();
    width = imageSize.width;
    height = imageSize.height;

    std::cout<<width<<" "<<height<<"\n";

    uc* ptrImageData = NULL;
    uc* ptrImageDataOut = NULL;

    cudaMalloc(&ptrImageDataOut, width * height * CHANNELS);
    cudaMalloc(&ptrImageData, width * height * CHANNELS);
    cudaMemcpy(ptrImageData, image1.data, width * height * CHANNELS, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(ptrImageDataOut, ptrImageData, width, height);
    cudaDeviceSynchronize(); // Esperar a que todos los bloques terminen

    Mat image2(height, width, CV_8UC3);
    cudaMemcpy(image2.data, ptrImageDataOut, width * height * CHANNELS, cudaMemcpyDeviceToHost);

    std::string nuevoNombre = argv[1];
    nuevoNombre = nuevoNombre.substr(0, nuevoNombre.find_last_of('.')) + "toGray.png";
    imwrite(nuevoNombre, image2);

    cudaFree(ptrImageData);
    cudaFree(ptrImageDataOut);

    namedWindow("Imagen Procesada", WINDOW_AUTOSIZE);
    imshow("Imagen Procesada", image2);
    waitKey(0);

    return 0;
}
