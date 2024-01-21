#include <GLFW/glfw3.h>
#include <H5Cpp.h>
#include <iostream>
#include <vector>

struct Point2D {
    float x, y;
};

std::vector<std::vector<Point2D>> readSkeletonData(const std::string& filename) {
    std::vector<std::vector<Point2D>> skeletonData;

    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("tracks");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, nullptr); // Get the dimensions of the dataset
        size_t numJoints = dims[0];  // Number of joints
        size_t numFrames = dims[1];  // Number of frames

        // Read the dataset
        std::vector<float> data(numJoints * numFrames * 3); // 3 coordinates per joint (x, y, z)
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

        // Fill skeletonData
        for (size_t i = 0; i < numFrames; ++i) {
            std::vector<Point2D> frame;
            for (size_t j = 0; j < numJoints; ++j) {
                size_t idx = (i * numJoints + j) * 3;
                Point2D point = {data[idx], data[idx + 1]}; // Only use x and y
                frame.push_back(point);
            }
            skeletonData.push_back(frame);
        }
    } catch (const H5::FileIException& e) {
        std::cerr << "Error reading HDF5 file: " << e.getDetailMsg() << std::endl;
    }

    return skeletonData;
}

// Function to draw the skeleton
void drawSkeleton(const std::vector<Point2D>& points) {
    glBegin(GL_LINES);
    for (size_t i = 0; i < points.size() - 1; ++i) {
        glVertex2f(points[i].x, points[i].y);
        glVertex2f(points[i + 1].x, points[i + 1].y);
    }
    glEnd();
}

// Function to process input
void processInput(GLFWwindow* window, bool& exit) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        exit = true;
    }
}

// Main function to initialize OpenGL and render skeleton
int main() {

    std::string filename = "predictions_analysis.h5";  
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Skeleton Visualization", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glOrtho(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);  // Orthographic projection (2D)

    // Example: assume locations are already populated from the h5 file
    // For each frame, `skeleton` contains the 2D points
    std::vector<Point2D> skeleton = {
        {100.0f, 100.0f}, {200.0f, 150.0f}, {300.0f, 100.0f}, // Example joints
        {400.0f, 150.0f}, {500.0f, 100.0f}
    };

    bool exit = false;
    while (!glfwWindowShouldClose(window) && !exit) {
        glClear(GL_COLOR_BUFFER_BIT);

        // Process input
        processInput(window, exit);

        // Draw skeleton (you can call it for each frame)
        drawSkeleton(skeleton);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
