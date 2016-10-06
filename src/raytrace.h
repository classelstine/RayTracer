#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

//include header file for glfw library so that we can use OpenGL
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define PI 3.1415926535

class Scene {
    public:
        vector eye_position;
        vector UL;
        vector UR;
        vector LL;
        vector LR;
        float resolution_x;
        float resolution_y;
        Scene();
        Scene(vector, vector, vector, vector, vector, float, float);
        void render();
}

Scene::Scene(vector eye, vector ul, vector ur, vector ll, vector lr, float rx, float ry) {
    eye_position = eye;
    UL = ul;
    UR = ur;
    LL = ll;
    LR = lr;
    resolution_x = rx;
    resolution_y = ry;
}

class Sample {
    public:
        float x;
        float y;
        Sample();
        Sample(float, float);
}

Sample::Sample(float my_x, float my_y) {
    x = my_x;
    y = my_y;
}

class Sampler {
    // It will generate (x,y) of a screen sample and return true.
    // Next time it gets called, it will generate another sample for the next pixel. It will return false when all the samples from all the pixels
    // are generated. (In our case, we generate 1 sample per pixel, at the pixel sample. Later on, if we want to do multi-sample per pixel, we need to modify this class.
    public:
        bool get_sample(Sample* sample);   
}

bool get_sample(Sample* sample);

class Matrix {
    public:
        float mat[4][4];
        Matrix();
}

class Film {
    vector<vector<Color>> pixels;
    public:
        void commit(Sample& sample, Color& color);
        void write_image();
        Film();
}

class Color {
    public:
        float r, g, b;
        Color(float, float, float);
        Color();
        void add_color(Color);
}

void Color::add_color(Color c) {
    r = r + c.r;
    g = g + c.g;
    b = b + c.b;
}

class Sphere {
    public:
        float radius;
        vector center;
}

