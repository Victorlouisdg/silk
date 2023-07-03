#include <iostream>
#include <Eigen/Core>

int main() {
    using Triangle = Eigen::Array3i;

    Triangle triangle0(0, 1, 2);

    std::cout << triangle0 << std::endl;
}