#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define BLUR_SIZE 3
#define CHANNELS 3

using namespace cv;

__global__
void colorToBlurConversion(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[(curRow * w + curCol) * CHANNELS];
                    pixels++;
                }
            }
        }

        out[(Row * w + Col) * CHANNELS] = (unsigned char)(pixVal / pixels);
        out[(Row * w + Col) * CHANNELS + 1] = (unsigned char)(pixVal / pixels);
        out[(Row * w + Col) * CHANNELS + 2] = (unsigned char)(pixVal / pixels);
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

    unsigned char* ptrImageData = NULL;
    unsigned char* ptrImageDataOut = NULL;

    cudaMalloc(&ptrImageDataOut, width * height * CHANNELS);
    cudaMalloc(&ptrImageData, width * height * CHANNELS);
    cudaMemcpy(ptrImageData, image1.data, width * height * CHANNELS, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    colorToBlurConversion<<<dimGrid, dimBlock>>>(ptrImageDataOut, ptrImageData, width, height);
    cudaDeviceSynchronize(); // Esperar a que todos los bloques terminen

    Mat image2(height, width, CV_8UC3);
    cudaMemcpy(image2.data, ptrImageDataOut, width * height * CHANNELS, cudaMemcpyDeviceToHost);

    std::string nuevoNombre = argv[1];
    nuevoNombre = nuevoNombre.substr(0, nuevoNombre.find_last_of('.')) + "_toBlur.jpeg";
    imwrite(nuevoNombre, image2);

    cudaFree(ptrImageData);
    cudaFree(ptrImageDataOut);

    namedWindow("Imagen Procesada", WINDOW_AUTOSIZE);
    imshow("Imagen Procesada", image2);
    waitKey(0);

    return 0;
}
