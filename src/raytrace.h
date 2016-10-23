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
        valarray<float> point;
        valarray<float> direction;
        Ray();
        Ray(valarray<float>,valarray<float>);
};

Ray::Ray(void) {
    point.resize(3);
    point = {0,0,-1};
    direction.resize(3);
    direction = {0,0,1};
}

Ray::Ray(valarray<float> p, valarray<float> d) {
    point = p;
    direction = d;
}

class Color {
    public:
        float r, g, b;
        Color(float, float, float);
        Color();
        void add_color(Color);
};

Color::Color(float red, float green, float blue) {
    r = red;
    g = green;
    b = blue;
}

Color::Color(void) {
    r = 0;
    g = 0;
    b = 0;
}

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

Sample::Sample(void) {
    x = 0;
    y = 0;
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

Raytracer::Raytracer(void) {
}

// Camera class, which can take a sample's coordinates and create a ray from the eye location through this point
// in the image. 
class Camera {
    valarray<float> eye_pos;
    public:
        Camera();
        void generate_ray(valarray<float> world_sample, Ray* cur_ray);
};

Camera::Camera(void) {
    eye_pos.resize(3);
    eye_pos = {0,0,-1};
}

// Scene will hold our Film, Camera
class Scene {
    Sampler sampler;
    Camera camera;
    Raytracer raytracer;
    Film film;
    public:
        valarray<float> eye_position;
        valarray<float> UL;
        valarray<float> UR;
        valarray<float> LL;
        valarray<float> LR;
        float resolution_x;
        float resolution_y;
        Scene();
        Scene(valarray<float>, valarray<float>, valarray<float>, valarray<float>, valarray<float>, float, float);
        void initialize();
        void render();
        void screen_to_world(valarray<float> screen, valarray<float>* world);
};

Scene::Scene(valarray<float> eye, valarray<float> ul, valarray<float> ur, valarray<float> ll, valarray<float> lr, float rx, float ry) {
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






