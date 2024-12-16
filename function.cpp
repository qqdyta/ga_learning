#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <algorithm>
#include "main.cu"
#include <numeric>

using namespace std;
using namespace cv;


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
