#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <valarray>

#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define PI 3.1415926535
using namespace std;

// Ray class represented as a point and a direction
class Ray {
    public:
        float px, py, pz, dx, dy, dz;
        Ray();
        Ray(float, float, float, float, float, float);
};

class Color {
    public:
        float r, g, b;
        Color(float, float, float);
        Color();
        void add_color(Color);
};

void Color::add_color(Color c) {
    r = r + c.r;
    g = g + c.g;
    b = b + c.b;
}
// Stores Screen Coordinate
class Sample {
    public:
        float x;
        float y;
        Sample();
        Sample(float, float);
};

Sample::Sample(float my_x, float my_y) {
    x = my_x;
    y = my_y;
}

class Sampler {
    // For now, the sampler just generates a single sample in the center of each pixel. 
    // It will generate (x,y) of a screen sample and return true.
    // Next time it gets called, it will generate another sample for the next pixel. It will return false when all the samples from all the pixels
    // are generated. (In our case, we generate 1 sample per pixel, at the pixel sample. Later on, if we want to do multi-sample per pixel, we need to modify this class.
    int current_px;
    int current_py;
    int max_x;
    int max_y;
    public:
        bool get_sample(Sample* sample);
        Sampler(int x_res, int y_res);
        Sampler();
};

//bool Sampler::get_sample(Sample* sample);

class Matrix {
    public:
        float mat[4][4];
        Matrix();
};

class Film {
    // The Film aggregates samples, storing them in buckets (one bucket for each pixel). When it's time to write out the file, the Film averages all the samples in each pixel's bucket to produce that pixel's output color. 
    // Each pixel is designated one bucket of Color vectors, once all samples are created, each bucket is averaged to create the color for each pixel.
    vector<vector<vector<Color>>> pixel_buckets;
       public:
        void commit(Sample sample, Color color);
        void write_image();
        Film(int, int, int);
        Film(void);
};

class Sphere {
    public:
        float radius;
        vector<float> center;
};

class Raytracer {
    public:
        void trace(Ray, Color*);
        Raytracer();
};

// Camera class, which can take a sample's coordinates and create a ray from the eye location through this point
// in the image. 
class Camera {
    public:
        Camera();
        void generate_ray(Sample cur_sample, Ray* cur_ray);
};
// Scene will hold our Film, Camera
class Scene {
    Sampler sampler;
    Camera camera;
    Raytracer raytracer;
    Film film;
    public:
        vector<float> eye_position;
        vector<float> UL;
        vector<float> UR;
        vector<float> LL;
        vector<float> LR;
        float resolution_x;
        float resolution_y;
        Scene();
        Scene(vector<float>, vector<float>, vector<float>, vector<float>, vector<float>, float, float);
        void initialize();
        void render();
};

Scene::Scene(vector<float> eye, vector<float> ul, vector<float> ur, vector<float> ll, vector<float> lr, float rx, float ry) {
    eye_position = eye;
    UL = ul;
    UR = ur;
    LL = ll;
    LR = lr;
    resolution_x = rx;
    resolution_y = ry;
    sampler = Sampler();
    camera = Camera();
    raytracer = Raytracer();
    film = Film(resolution_x, resolution_y, 1);
}






