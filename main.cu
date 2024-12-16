#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>


using namespace cv;

struct t_point {
    int x;
    int y;
};

struct Triangle {
    t_point ONE;
    t_point TWO;
    t_point THR;
    unsigned char R;
    unsigned char G;
    unsigned char B;
    unsigned char A; // Alpha channel
};

// 变异函数
void mutate(std::vector<Triangle>& individual, int width, int height);

// 杂交函数
std::vector<Triangle> crossover(const std::vector<Triangle>& parent1, const std::vector<Triangle>& parent2);

// 筛选和淘汰函数
std::vector<std::vector<Triangle>> selectAndCull(const std::vector<std::vector<Triangle>>& population, const std::vector<float>& fitnesses, int numSurvivors);

// 进化函数
std::vector<std::vector<Triangle>> evolvePopulation(std::vector<std::vector<Triangle>>& population, Mat& targetImage, int width, int height, int numSurvivors, int numIterations)


__global__ void drawTriangles(Triangle* triangles, unsigned char* output, int width, int height, int numTriangles);

std::vector<Vec3b> createBlankImageArray(int width, int height) {
    std::vector<Vec3b> rgbData(width * height, Vec3b(0, 0, 0));
    return rgbData;
}

Triangle generateRandomTriangle(int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDis(0, width - 1);
    std::uniform_int_distribution<> yDis(0, height - 1);
    std::uniform_int_distribution<> colorDis(0, 255);
    Triangle triangle;
    triangle.ONE.x = xDis(gen);
    triangle.ONE.y = yDis(gen);
    triangle.TWO.x = xDis(gen);
    triangle.TWO.y = yDis(gen);
    triangle.THR.x = xDis(gen);
    triangle.THR.y = yDis(gen);
    triangle.R = colorDis(gen);
    triangle.G = colorDis(gen);
    triangle.B = colorDis(gen);
    triangle.A = colorDis(gen);
    return triangle;
}

std::vector<std::vector<Triangle>> initializePopulation(int populationSize, int numTriangles, int width, int height) {
    std::vector<std::vector<Triangle>> population(populationSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> seedDis(0, 100000);

    for (int i = 0; i < populationSize; ++i) {
        std::vector<Triangle> individual;
        for (int j = 0; j < numTriangles; ++j) {
            individual.push_back(generateRandomTriangle(width, height));
        }
        population[i] = individual;
    }
    return population;
}

Triangle generateRandomTriangle(int width, int height, std::mt19937& gen) {
    std::uniform_int_distribution<> xDis(0, width - 1);
    std::uniform_int_distribution<> yDis(0, height - 1);
    std::uniform_int_distribution<> colorDis(0, 255);
    Triangle triangle;
    triangle.ONE.x = xDis(gen);
    triangle.ONE.y = yDis(gen);
    triangle.TWO.x = xDis(gen);
    triangle.TWO.y = yDis(gen);
    triangle.THR.x = xDis(gen);
    triangle.THR.y = yDis(gen);
    triangle.R = colorDis(gen);
    triangle.G = colorDis(gen);
    triangle.B = colorDis(gen);
    triangle.A = colorDis(gen);
    return triangle;
}


__global__ void calculateMSE(unsigned char* target, unsigned char* candidate, float* mse, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float diffR = target[idx * 4 + 0] - candidate[idx * 4 + 0];
    float diffG = target[idx * 4 + 1] - candidate[idx * 4 + 1];
    float diffB = target[idx * 4 + 2] - candidate[idx * 4 + 2];

    atomicAdd(mse, (diffR * diffR + diffG * diffG + diffB * diffB) / 3.0f);
}


float evaluateIndividual(const Mat& targetImage, const unsigned char* candidate, int width, int height) {

    // 分配设备内存来储存目标图像
    unsigned char* d_target;
    cudaMalloc(&d_target, width * height * 4 * sizeof(unsigned char));
    cudaMemcpy(d_target, targetImage.data, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 分配设备内存
    unsigned char* d_candidate;
    cudaMalloc(&d_candidate, width * height * 4 * sizeof(unsigned char));
    cudaMemcpy(d_candidate, candidate, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 分配设备内存来储存MES结果
    float* d_mse;
    cudaMalloc(&d_mse, sizeof(float));
    cudaMemset(d_mse, 0, sizeof(float));

    //设置网格和维度
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    calculateMSE<<<gridSize, blockSize>>>(d_target, d_candidate, d_mse, width, height);

    float mse;
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_target);
    cudaFree(d_candidate);
    cudaFree(d_mse);

    return mse / (width * height);
}

int main() {
    Mat image = imread("D:\\tmp\\cuda\\a.jpg", IMREAD_COLOR);

    const int TARGET_IMAGE_WIDTH = image.cols;
    const int TARGET_IMAGE_HEIGHT = image.rows;

    // Initialize population
    std::vector<std::vector<Triangle>> population = initializePopulation(64, 32, TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT);

    // Allocate host memory
    Triangle* h_triangles = new Triangle[64 * 32];
    unsigned char* h_output = new unsigned char[TARGET_IMAGE_WIDTH * TARGET_IMAGE_HEIGHT * 4];

    // Flatten the population into a single array
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 32; ++j) {
            h_triangles[i * 32 + j] = population[i][j];
        }
    }

    // Allocate device memory
    Triangle* d_triangles;
    unsigned char* d_output;
    cudaMalloc(&d_triangles, sizeof(Triangle) * 64 * 32);
    cudaMalloc(&d_output, TARGET_IMAGE_WIDTH * TARGET_IMAGE_HEIGHT * 4);

    // Copy data from host to device
    cudaMemcpy(d_triangles, h_triangles, sizeof(Triangle) * 64 * 32, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((TARGET_IMAGE_WIDTH + blockSize.x - 1) / blockSize.x, (TARGET_IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    drawTriangles<<<gridSize, blockSize>>>(d_triangles, d_output, TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 32);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, TARGET_IMAGE_WIDTH * TARGET_IMAGE_HEIGHT * 4, cudaMemcpyDeviceToHost);

    // Create a new blank image
    Mat resultImage(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, CV_8UC4);
    for (int y = 0; y < TARGET_IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < TARGET_IMAGE_WIDTH; ++x) {
            int idx = (y * TARGET_IMAGE_WIDTH + x) * 4;
            resultImage.at<Vec4b>(y, x) = Vec4b(h_output[idx + 0], h_output[idx + 1], h_output[idx + 2], h_output[idx + 3]);
        }
    }



    // Display the result image
    namedWindow("Result Image", WINDOW_NORMAL);
    imshow("Result Image", resultImage);
    waitKey(0);

    // Clean up resources
    delete[] h_triangles;
    delete[] h_output;
    cudaFree(d_triangles);
    cudaFree(d_output);

    return 0;
}

__global__ void drawTriangles(Triangle* triangles, unsigned char* output, int width, int height, int numTriangles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char r = 0, g = 0, b = 0, a = 0;

    for (int i = 0; i < numTriangles; ++i) {
        float ax = triangles[i].ONE.x, ay = triangles[i].ONE.y;
        float bx = triangles[i].TWO.x, by = triangles[i].TWO.y;
        float cx = triangles[i].THR.x, cy = triangles[i].THR.y;

        // Check if the point is inside the triangle
        bool inside = ((x - ax) * (by - ay) - (y - ay) * (bx - ax)) > 0 &&
                      ((x - bx) * (cy - by) - (y - by) * (cx - bx)) > 0 &&
                      ((x - cx) * (ay - cy) - (y - cy) * (ax - cx)) > 0;

        if (inside) {
            unsigned char tr = triangles[i].R, tg = triangles[i].G, tb = triangles[i].B, ta = triangles[i].A;
            // Simple alpha blending
            r = (tr * ta / 255) + (r * (255 - ta) / 255);
            g = (tg * ta / 255) + (g * (255 - ta) / 255);
            b = (tb * ta / 255) + (b * (255 - ta) / 255);
            a += ta;
            a = min(a, 255);  // Ensure it does not exceed 255
        }
    }

    output[idx * 4 + 0] = r;  // R
    output[idx * 4 + 1] = g;  // G
    output[idx * 4 + 2] = b;  // B
    output[idx * 4 + 3] = a;  // A
}


std::vector<std::vector<Triangle>> selectAndCull(const std::vector<std::vector<Triangle>>& population, const std::vector<float>& fitnesses, int numSurvivors) {
    // 创建一个索引向量
    std::vector<int> indices(population.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 根据适应度对索引进行排序
    std::sort(indices.begin(), indices.end(), [&fitnesses](int a, int b) {
        return fitnesses[a] < fitnesses[b];
    });

    // 选择适应度最高的numSurvivors个个体
    std::vector<std::vector<Triangle>> survivors(numSurvivors);
    for (int i = 0; i < numSurvivors; ++i) {
        survivors[i] = population[indices[i]];
    }

    return survivors;
}

// 杂交函数
std::vector<Triangle> crossover(const std::vector<Triangle>& parent1, const std::vector<Triangle>& parent2) {
    std::vector<Triangle> child(parent1.size());
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (rand() % 2 == 0) {
            child[i] = parent1[i];
        } else {
            child[i] = parent2[i];
        }
    }
    return child;
}

// 变异函数
void mutate(std::vector<Triangle>& individual, int width, int height) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> indexDis(0, individual.size() - 1);
    std::uniform_int_distribution<> coordDis(0, 1); // 0 for x, 1 for y
    std::uniform_int_distribution<> colorDis(0, 255);
    std::uniform_int_distribution<> pointDis(0, 2); // 0 for ONE, 1 for TWO, 2 for THR

    // 选择一个随机三角形进行变异
    int index = indexDis(gen);
    int point = pointDis(gen);
    int coord = coordDis(gen);

    // 变异坐标
    if (coord == 0) { // x coordinate
        individual[index].ONE.x = rand() % width;
        individual[index].TWO.x = rand() % width;
        individual[index].THR.x = rand() % width;
    } else { // y coordinate
        individual[index].ONE.y = rand() % height;
        individual[index].TWO.y = rand() % height;
        individual[index].THR.y = rand() % height;
    }

    // 变异颜色
    individual[index].R = colorDis(gen);
    individual[index].G = colorDis(gen);
    individual[index].B = colorDis(gen);
    individual[index].A = colorDis(gen);
}
