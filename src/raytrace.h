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

//dot valarrays
float dot(valarray<float> v1, valarray<float> v2) {
    float d = (v1 * v2).sum();
    return d;
}

//cross valarrays
void cross(valarray<float> v1, valarray<float>  v2, valarray<float> *v3)  {
    *v3 = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
    /*
    v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
    */
}

void dist(valarray<float> p1, valarray<float> p2, float* d) {
    valarray<float> v = p1 - p2;
    valarray<float> tmp = pow(v, 2);
    float magnitude = sqrt(tmp.sum());
    *d = magnitude;
}

//normalize valarrays
void normalize(valarray<float> *v) { 
    valarray<float> p = *v;
    valarray<float> tmp = pow(p, 2);
    float magnitude = sqrt(tmp.sum());
    p /= magnitude; 
    *v = p;
    //*v = (*v)*(1/magnitude);
} 

/*
 *                    +===========+
 *                    | RAY CLASS |
 *                    +===========+
 *
 * A parametrized ray defined by a point and a direction. 
 * Both are represented by valarrays for element-wise 
 * calculations. 
 *
 */
class Ray {
    public:
        valarray<float> point;
        valarray<float> direction;
        Ray();
        Ray(valarray<float>,valarray<float>);
        void eval(float t, valarray<float>* cord);
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

/*
 *                    +=============+
 *                    | COLOR CLASS |
 *                    +=============+
 *
 *  Representation of color with three floats for r,g,b. 
 *  The floats are between 0 and 1. We define various 
 *  operations for the color class like add, multiply, 
 *  and scale so that we can use our shading models.
 *
 */

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

void mult_color(Color c1, Color c2, Color *c3) {
    c3->r = c1.r * c2.r;
    c3->g = c1.g * c2.g;
    c3->b = c1.b * c2.b;
} 

void scale_color(float c, Color c1, Color *c2) {
    c2->r = c * c1.r;
    c2->g = c * c1.g;
    c2->b = c * c1.b;
}

/*
 *                    +==============+
 *                    | SAMPLE CLASS |
 *                    +==============+
 *
 * It holds a sample that represents our current screen coordinate. 
 * Implemented to interact with SAMPLER CLASS 
 *
 */
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

/*
 *                    +===============+
 *                    | SAMPLER CLASS |
 *                    +===============+
 *
 * For now, the sampler just generates a single sample in the center of each pixel. 
 * It will generate (x,y) of a screen sample and return true. Next time it gets called, 
 * it will generate another sample for the next pixel. It will return false when all the
 * samples from all the pixels are generated. In our case, we generate 1 sample per pixel, 
 * at the pixel sample. Later on, if we want to do multi-sample per pixel, we need to 
 * modify this class.
 *
 */

class Sampler {
    int current_px;
    int current_py;
    int max_x;
    int max_y;
    public:
        bool get_sample(Sample* sample);
        Sampler(int x_res, int y_res);
        Sampler();
};

/*
 *                    +============+
 *                    | FILM CLASS |
 *                    +============+
 *
 * The film class holds our data structure where we store our image.
 * It's main functions are commit and write image. Commit assigns a 
 * pixel a color by pushing it back to a vector of existing colors. 
 * When we write the image we average over each pixel bucket and use 
 * an opensource header file to save the image.
 *
 */

class Film {
    vector<vector<vector<Color>>> pixel_buckets;
    int res_x;
    int res_y;
    int sample_rate;
       public:
        void commit(Sample sample, Color color);
        void write_image();
        Film(int, int, int);
        Film(void);
};

/*                  +================+
 *                  | MATERIAL CLASS |
 *                  +================+
 *
 *  This class is a representation of material behaviors, such as KA, KD, KS, 
 *  SPU, SPV, refraction, transparency, and reflection. All resulting in the
 *  way an object will end up looking (glossy, rough, metallic, etc)
 *
*/

class Material {
    public:
        Color KA, KD, KS;
        float SPU, SPV;
        Material(Color, Color, Color, float, float);
        Material();
};

Material::Material(Color ka, Color kd, Color ks, float spu, float spv) {
    KA = ka;
    KD = kd;
    KS = ks;
    SPU = spu;
    SPV = spv;
}

Material::Material(void) {
    KA = Color(0.5, 0.5, 0.5);
    KD = Color(0.5, 0.5, 0.5);
    KS = Color(0.5, 0.5, 0.5);
    SPU = 2;
    SPV = 2;
}

/*
 *                    +==============+
 *                    | OBJECT CLASS |
 *                    +==============+
 *
 *  This class is a representation of all the objects we are going to hold 
 *  in our scene. Right now we have implemented the sphere and triangle classes.
 *  Sphere is defined by its center and radius. 
 *  Trinagle is defined by its three vertices. 
 *  Each object has a unique implementation of t_hit, which sees if our
 *  traced ray intersects with the object. We get these implementations from Shirley.
 *  Each object also implements get_normal, which we use for shading and intersection
 *  calcultions.
 *  Each object has a KA, KD, KS, SPU, SPV value, which we use for shading. 
 */

class Object {
    public: 
        Material material;
        virtual bool t_hit(Ray ray, float* t) { cout << "WRONG FUNCTION" << endl; return false; }
        virtual void get_normal(valarray<float> point, valarray<float>* normal);
};


void Object::get_normal(valarray<float> point, valarray<float>* normal) {
}

class Sphere: public Object {
    public:
        Sphere(valarray<float> c, float r, Material);
        float radius;
        valarray<float> center;
        bool t_hit(Ray ray, float* t);
        virtual void get_normal(valarray<float> point, valarray<float>* normal);
};

Sphere::Sphere(valarray<float> c, float r, Material m) {
    radius = r;
    center = c;
    material = m;
}

void Sphere::get_normal(valarray<float> p, valarray<float>* n) {
    valarray<float> normal = p - center;
    normalize(&normal);
    n->swap(normal);
}

// Triangle class holds three x,y,z cordinates defining a triangle 
class Triangle : public Object {
    valarray<float> p1;
    valarray<float> p2;
    valarray<float> p3;
    valarray<float> normal;
    public:
        Triangle(valarray<float>, valarray<float>, valarray<float>, Material);
        virtual bool t_hit(Ray ray, float* t);
        virtual void get_normal(valarray<float> point, valarray<float>* normal);
};

Triangle::Triangle(valarray<float> one, valarray<float> two, valarray<float> three, Material m) {
    material = m;
    p1 = one;
    p2 = two;
    p3 = three;
    cout << "creating triangle" << endl;
    normal = {0,0,0};
    cross(three-two, three-one, &normal);
    cout << "cross done" << endl;
    cout << "nomral vector " << normal[0] << " " << normal[1] << " " << normal[2] << endl;
    normalize(&normal);
    cout << "end create triangle" << endl;
}

void Triangle::get_normal(valarray<float> p, valarray<float>* n) {
    *n = {normal[0],normal[1],normal[2]};
}
// NEEDS TO BE IMPLEMENTED

/*
 *                    +=============+
 *                    | LIGHT CLASS |
 *                    +=============+
 *
 * Representation of a light, which is used for our shading model. We
 * have directional and point lights. 
 * A point light is located at xyz. 
 * A point light is coming at a direction of xyz.
 * We implement light_vector, which returns the vector from our point to 
 * the light source. 
 *
 */

class Light {
    public:
        bool is_direct;
        valarray<float> xyz;
        Color color;
        Light(valarray<float> p, Color c, bool is_d);
        void light_vector(valarray<float> point, valarray<float> *l_vec);
};

Light::Light(valarray<float> p, Color c, bool is_d) {
    xyz = p;
    color = c;
    is_direct = is_d;
    }

void Light::light_vector(valarray<float> point, valarray<float> *l_vec) {
    if (is_direct) {
        *l_vec = -1*xyz;
        normalize(l_vec);
    } else {
        *l_vec = xyz - point;
        normalize(l_vec);
    }
}

/*
 *                    +==============+
 *                    | SHADER CLASS |
 *                    +==============+
 *
 * This holds the light environment of our scene. Lights is a 
 * list of lights that is currently in our environment. We can 
 * call phong on an object, normal, point and view to be able 
 * to calculate the resulting color. 
 *
 */

class Shader {
    vector<Light> lights;
    public :
        Shader();
        Shader(vector<Light>);
        void phong(valarray<float> point, valarray<float> normal, valarray<float> view, Color * c, Material *mat);
};

Shader::Shader(void) {
    /*
    valarray<float> p1 = {10, 10, 0};
    Color color(1.0, 1.0, 1.0);
    Light light1 = Light(p1, color, false);
    */
    valarray<float> p1 = {10, 0, 0};
    valarray<float> p2 = {0, 10, 0};
    Color color(1.0, 1.0, 0.0);
    Color color1(0.0, 1.0, 1.0);
    Light light1 = Light(p1, color, false);
    Light light2 = Light(p2, color1, false);
    //lights = {light1};
    lights = {light1,light2};
}

Shader::Shader(vector<Light> l_list) {
    lights = l_list;
}

void reflectance(valarray<float> light_source, valarray<float> normal, valarray<float> *reflectance);

float find_specular_power(valarray<float> normal, valarray<float> view, valarray<float> light_vec, Material *obj);

/*
 *                    +=================+
 *                    | RAYTRACER CLASS |
 *                    +=================+
 *
 * Raytracer traces a ray - this is its main function. The raytracer takes
 * a ray and knows a shading a environment. It sends the ray out through our
 * view window and sees if it intersects with any object if it does interesect 
 * it uses the shader to compute the resulting shader value. 
 *
 */

class Raytracer {
    bool is_obj;
    Shader shader;
    public:
        void trace(Ray, Color*);
        Raytracer();
};

Raytracer::Raytracer(void) {
   is_obj = true;
   shader = Shader();
}

/*
 *                    +==============+
 *                    | CAMERA CLASS |
 *                    +==============+
 *
 * Camera class, which can take a sample's coordinates and create a ray from the eye 
 * location through this point in the image. It holds the eye position of the 
 * scene and given any sample it generates a ray from the eye postion through the 
 * xyz coordinate of the sample.
 *
 */

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

/*
 *                    +=============+
 *                    | SCENE CLASS |
 *                    +=============+
 *
 * Given an eye position and image plane coordinates holds corresponding sampler, 
 * camera, raytracer and film. It's most important function is render, which uses
 * the sampler iterator to generate the color values for every pixel in our view 
 * window and then uses the film to write the image to a png file.
 *
 */

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






