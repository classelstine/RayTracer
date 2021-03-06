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
        Color KA, KD, KS, KR;
        float SPU, SPV;
        Material(Color, Color, Color, Color, float, float);
        Material();
};

Material::Material(Color ka, Color kd, Color ks, Color kr, float spu, float spv) {
    KA = ka;
    KD = kd;
    KS = ks;
    KR = kr;
    SPU = spu;
    SPV = spv;
}

Material::Material(void) {
    KA = Color(.8, 0.2, 0.0);
    KD = Color(0.7, 0.2, 0.0);
    KS = Color(0.7, 0.2, 0.0);
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

void translate(float x, float y, float z, valarray<float> world, valarray<float> *obj) {
    *obj = {world[0] - x, world[1] - y, world[2] - z};    
}

void scale(float x, float y, float z, valarray<float> world, valarray<float> *obj) {
    *obj = {world[0]/x, world[1]/y, world[2]/z};    
}

void deg_to_rad(float d, float* r) {
    *r = (PI / 180.0) * d;
}

void rotate(float x, float y, float z, float radians, valarray<float> world, valarray<float> *obj) {
    valarray<float> axis = {x, y, z};
    normalize(&axis);
    float ux = axis[0];
    float uy = axis[1];
    float uz = axis[2];

    float c = cos(radians);
    float s = sin(radians);

    valarray<float> row1 = {c+(pow(ux, 2)*(1-c)), (ux*uy*(1-c)-uz*s), (ux*uz*(1-c)+uy*s)};
    valarray<float> row2 = {(ux*uy*(1-c)+uz*s), c+(pow(uy, 2)*(1-c)), (uy*uz*(1-c)-ux*s)};
    valarray<float> row3 = {(ux*uz*(1-c)-uy*s)  ,(uy*uz*(1-c)+ux*s) , c+(pow(uz, 2)*(1-c))};

    *obj = { (row1*world).sum(), (row2*world).sum(), (row3*world).sum()};    
}

void exp_rotate(float x, float y, float z, valarray<float> world, valarray<float>* obj) {
    float x_rad = 0.0;
    float y_rad = 0.0;
    float z_rad = 0.0;
    deg_to_rad(x, &x_rad);
    deg_to_rad(y, &y_rad);
    deg_to_rad(z, &z_rad);
    rotate(1.0, 0.0, 0.0, x_rad, world, obj);
    rotate(0.0, 1.0, 0.0, y_rad, *obj, obj);
    rotate(0.0, 0.0, 1.0, z_rad, *obj, obj);
}


class Object {
    public: 
        Material material;
        vector<valarray<float>> lin_transform;
        void world_to_obj(valarray<float> world, valarray<float>* obj);
        virtual void dist(valarray<float>, float*);
        virtual bool t_hit(Ray ray, float* t) { cout << "WRONG FUNCTION" << endl; return false; }
        virtual void get_normal(valarray<float> point, valarray<float>* normal);
        //void set_translation(float, float, float);
        //void set_rotation(float, float, float);
        //void set_scale(float, float, float); 
};
/*
void Object::set_translation(float a, float b, float c) {
    trans = {a, b, c};
}

void Object::set_rotation(float a, float b, float c) {
    rot = {a, b, c};
}

void Object::set_scale(float a, float b, float c) {
    sca = {a, b, c};
}
*/

void Object::world_to_obj(valarray<float> w, valarray<float>* o) {
    for (valarray<float> inst : lin_transform) {
        float i = inst[0];
        float x = inst[1];
        float y = inst[2];
        float z = inst[3];
        if (i== 0) {
            exp_rotate(x,y,z,w,&w);
        } else if (i == 1) {
            scale(x,y,z,w,&w);
        } else {
            translate(x,y,z,w,&w);
        }
    }
    *o = {w[0], w[1], w[2]};
}

void Object::dist(valarray<float> cur_pt,float* d) {
}


void Object::get_normal(valarray<float> point, valarray<float>* normal) { 
} 


class Func_Sphere: public Object {
    public:
        Func_Sphere(valarray<float> c, float r, Material);
        bool t_hit(Ray ray, float* t);
        void get_normal(valarray<float> point, valarray<float>* normal);
        void dist(valarray<float> point, float *d);
        valarray<float> center;
        float radius;
        float mode;
        //void translate(float, float, float, valarray<float>, valarray<float>*);
        //void scale(float, float, float, valarray<float>, valarray<float>*);
        //void rotate(float, float, float, float, valarray<float>, valarray<float>*);
        //void exp_rotate(float, float, float, valarray<float>, valarray<float>*);
};

Func_Sphere::Func_Sphere(valarray<float> c, float r, Material m) {
    center = c;
    radius = r;
    material = m;
    lin_transform = {};
    mode = 0;
    //cout << "func created r:" << r << endl;
}


// If you want a transformation, you must change p before passing into this function. 
void Func_Sphere::dist(valarray<float> p, float*d) {
    world_to_obj(p, &p);
    
    //float dtc = sqrt(pow((p - center), 2).sum());
    //if (abs(p[0]) <= 0.01 && abs(p[1]) <= 0.01) {
//cout << "P" << p[0] << "," << p[1] << "," <<  p[2] << endl;
    //cout << "D: " << dtc - radius << endl;
    //}
 
    //*d = radius - dtc;

    // This part implements an ellipse.
    /* 
    float r = sqrt(pow(p[0], 2) + pow(p[1], 2));
    p[0] = p[0]/(1.0 - 0.2 * r);
    */
    // This part makes a grid of spheres.
    //p[0] = 2.0 * ((p[0]/2.0) - floor((p[0]/2.0) + (1.0/2.0)));
    //p[1] = 2.0 * ((p[1]/2.0) - floor((p[1]/2.0) + (1.0/2.0)));
    //p[2] = 2.0 * ((p[2]/2.0) - floor((p[2]/2.0) + (1.0/2.0)));
    
    //translate(2.0, 2.0, 2.0, p, &p);
    //translate(2.0, 2.0, 2.0, center, &center);
    // THIS IS THE MOST BASIC FORM OF A SPHERE
    //translate(2.0, 2.0, 0.0, p, &p);
    //float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
    //valarray<float> c2 = {2.0, 2.0, 7.0};
    //float dtc2 = (sqrt(pow((p - c2), 2).sum()) - 3.0);
    //float inf_spheres = (-1) * (dist_to_center - radius);
    //float plane = 5.0 - p[1];
    //float plane2 = 7.0 - p[0];
    //float plane3 = 7.0 + p[0];
    //float plane4 = 5.0 + p[1];
    //float plane5 = 11.0 - p[2];
    //float plane6 = 2.0 + p[2];
    //*d = min(min(min(min(min(min(min(plane6, plane5), plane4), plane3), plane2), plane),  dist_to_center), dtc2);
    //*d = min(min(min(min(min(plane6, plane4), plane3), plane2), plane),  dist_to_center);
    //*d = dist_to_center;
    //float z = 20.0 + p[2];
    //*d = min(dist_to_center, z);
    
    // THIS IS MULTIPLE SPHERES
    /*
    valarray<float> c2 = {-6.0, 0.0, 15.0};
    valarray<float> c3 = {0.0, 6.0, 15.0};
    valarray<float> c4 = {6.0, 0.0, 15.0};
    valarray<float> c5 = {0.0, -6.0, 15.0};
    float dtc1 = sqrt(pow((p - center), 2).sum());
    scale(1.0,2.0,1.0, p, &p);
    float dtc2 = sqrt(pow((p - c2), 2).sum());
    scale(1.5,0.5,1.0,p,&p);
    float dtc3 = sqrt(pow((p - c3), 2).sum());
    scale(0.666,2.0,1.0, p, &p);
    float dtc4 = sqrt(pow((p - c4), 2).sum());
    scale(1.5,0.5,1.0,p,&p);
    float dtc5 = sqrt(pow((p - c5), 2).sum());

    float d1 = dtc1 - radius;
    float d2 = dtc2 - 2;
    float d3 = dtc3 - 2;
    float d4 = dtc4 - 2;
    float d5 = dtc5 - 2;
    *d = min(min(min(min(d1, d2), d3), d4), d5);
    */
    //cout << *d<< endl;

    // THIS IS MULTIPLE SPHERES
    /*
    valarray<float> c2 = { -2.0, -2.0, 20.0};
    float dtc1 = sqrt(pow((p - center), 2).sum());
    float dtc2 = sqrt(pow((p - c2), 2).sum());
    float d1 = dtc1 - radius;
    float d2 = dtc2 - 0.5;
    *d = min(d1, d2);
    */

    // THIS IS A PLANE AND A SPHERE
    /*
    float dist_to_center = sqrt(pow((p - center), 2).sum()); 
    float plane = 3.0 - p[0];
    *d = min(dist_to_center - radius, plane);
    */



    // THIS IS A TRANSLATION
    /*
    valarray<float> obj_point = {0,0,0};
    translate(0, 0, 20, p, &obj_point);
    float dist_to_center = sqrt(pow((obj_point), 2).sum());
    float plane = 3.0 - obj_point[0];
    *d = min(dist_to_center - radius, plane);
    */

    // THIS IS A LINEAR TRANSLATION
    /*
    valarray<float> obj_point = {0,0,0};
    float dist_to_center = sqrt(pow((obj_point), 2).sum());
    float plane = 3.0 - obj_point[0];
    *d = min(dist_to_center - radius, plane);
    */
    // THIS IS A SUPERQUADRIC
    /*
    float r = 0.5;
    float s = 0.5;
    float t = 0.5;
    valarray<float> obj_point = {0,0,0};
    rotate(1.0,0,1.0,PI/6,p,&obj_point);
    scale(10, 6, 6, obj_point, &obj_point);
    translate(0,0,10,obj_point,&obj_point);
    *d = (-1) *(1.0 - pow(abs(obj_point[0]),r) - pow(abs(obj_point[1]), s)- pow(abs(obj_point[2]), t));
    */
    // rainbow ball on surface -- 1
    if (mode == 1) {
        float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
        float plane = 10.0 - p[2];
        *d = min(dist_to_center, plane);
    }
    // rainbow box -- 2 and 3
    if (mode == 2 || mode == 3) {
        float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
        float plane = 2.0 - p[1];
        float plane2 = 2.0 - p[0];
        float plane3 = 2.0 + p[0];
        float plane4 = 2.0 + p[1];
        float plane5 = 8.0 - p[2];
        float plane6 = 2.0 + p[2];
        *d = min(min(min(min(min(min(plane6, plane5), plane4), plane3), plane2), plane),  dist_to_center);
    }  
    // reflective spheres -- 4 
    if (mode == 4) {
        
        valarray<float> c2 = {-6.0, 0.0, 15.0};
        valarray<float> c3 = {0.0, 6.0, 15.0};
        valarray<float> c4 = {6.0, 0.0, 15.0};
        valarray<float> c5 = {0.0, -6.0, 15.0};
        float dtc1 = sqrt(pow((p - center), 2).sum());
        float dtc2 = sqrt(pow((p - c2), 2).sum());
        float dtc3 = sqrt(pow((p - c3), 2).sum());
        float dtc4 = sqrt(pow((p - c4), 2).sum());
        float dtc5 = sqrt(pow((p - c5), 2).sum());

        float d1 = dtc1 - radius;
        float d2 = dtc2 - 2;
        float d3 = dtc3 - 2;
        float d4 = dtc4 - 2;
        float d5 = dtc5 - 2;
        *d = min(min(min(min(d1, d2), d3), d4), d5);

    }
    // reflective ellipses -- 5 
    if (mode == 5) {  
        valarray<float> c2 = {-6.0, 0.0, 15.0};
        valarray<float> c3 = {0.0, 6.0, 15.0};
        valarray<float> c4 = {6.0, 0.0, 15.0};
        valarray<float> c5 = {0.0, -6.0, 15.0};
        float dtc1 = sqrt(pow((p - center), 2).sum());
        scale(1.0,2.0,1.0, p, &p);
        float dtc2 = sqrt(pow((p - c2), 2).sum());
        scale(1.5,0.5,1.0,p,&p);
        float dtc3 = sqrt(pow((p - c3), 2).sum());
        scale(0.666,2.0,1.0, p, &p);
        float dtc4 = sqrt(pow((p - c4), 2).sum());
        scale(1.5,0.5,1.0,p,&p);
        float dtc5 = sqrt(pow((p - c5), 2).sum());

        float d1 = dtc1 - radius;
        float d2 = dtc2 - 2;
        float d3 = dtc3 - 2;
        float d4 = dtc4 - 2;
        float d5 = dtc5 - 2;
        *d = min(min(min(min(d1, d2), d3), d4), d5);

    }

    // x and y blue balls -- 6
    if (mode == 6) {    
        p[0] = 2.0 * ((p[0]/2.0) - floor((p[0]/2.0) + (1.0/2.0)));
        p[1] = 2.0 * ((p[1]/2.0) - floor((p[1]/2.0) + (1.0/2.0)));

        float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
        *d = dist_to_center;
    }

    // x and y yellow balls -- 7
    if (mode == 7) {    
        p[0] = 2.0 * ((p[0]/2.0) - floor((p[0]/2.0) + (1.0/2.0)));
        p[1] = 2.0 * ((p[1]/2.0) - floor((p[1]/2.0) + (1.0/2.0)));

        float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
        *d = dist_to_center;
    }

    //YELLOW BALLS 
    //TWO BALLS ONE BOX -- 8
    if(mode == 8) { 
        float dist_to_center = (sqrt(pow((p - center), 2).sum()) - radius);
        valarray<float> c2 = {2.0, 2.0, 7.0};
        float dtc2 = (sqrt(pow((p - c2), 2).sum()) - 3.0);
        float plane = 5.0 - p[1];
        float plane2 = 7.0 - p[0];
        float plane3 = 7.0 + p[0];
        float plane4 = 5.0 + p[1];
        float plane5 = 11.0 - p[2];
        float plane6 = 2.0 + p[2];
        *d = min(min(min(min(min(min(min(plane6, plane5), plane4), plane3), plane2), plane),  dist_to_center), dtc2);
    }
    
}

bool Func_Sphere::t_hit(Ray ray, float* t) {
    float epsilon = 0.01;
    float cur_t = 0;
    valarray<float> cur_pt = {0.0, 0.0, 0.0};
    float d = 1000;
    float max_val = 1000;
    while(cur_t < max_val) {
        ray.eval(cur_t, &cur_pt);
        dist(cur_pt, &d);
        if (d < epsilon) {
            *t = cur_t;
            if (false) {
            cout << "Ray: " << ray.direction[0] <<"," << ray.direction[1] << "," << ray.direction[2] << endl;
            cout << "T FOUND: " << *t << endl;
            cout << "D found: " << d << endl;
            }
            return true;
        }
        cur_t = cur_t + d;
    }
    return false;

}

void Func_Sphere::get_normal(valarray<float> point, valarray<float>* normal) {
    /*
    valarray<float> n = point - center;
    normalize(&n);
    normal->swap(n);
    */
    world_to_obj(point, &point);
    valarray<float> n = {0.0,0.0,0.0};
    valarray<float> e1 = {0.01, 0.0, 0.0};
    valarray<float> e2 = {0.0, 0.01, 0.0};
    valarray<float> e3 = {0.0, 0.0, 0.01};
    float a, b, c, d, e, f;
    dist(point + e1, &a);
    dist(point - e1, &b);
    dist(point + e2, &c);
    dist(point - e2, &d);
    dist(point + e3, &e);
    dist(point - e3, &f);
    n[0] = a - b;
    n[1] = c - d;
    n[2] = e - f;
    normalize(&n);
    normal->swap(n);

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
    lin_transform = {};
}

void Sphere::get_normal(valarray<float> p, valarray<float>* n) {
    world_to_obj(p, &p);
    valarray<float> c2 = {0.0,0.0,0.0};
    world_to_obj(center, &c2);
    valarray<float> normal = p - c2;
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
    lin_transform = {};
    normal = {0,0,0};
    cross(three-two, three-one, &normal);
    normalize(&normal);
}

/*
void Triangle::set_translation(float a, float b, float c) {
    trans = {a, b, c};
    translate(a, b, c, p1, &p1);
    translate(a, b, c, p2, &p2);
    translate(a, b, c, p3, &p3);
    normal = {0,0,0};
    cross(p3 - p2, p3 - p1, &normal);
    normalize(&normal);
}

void Triangle::set_rotation(float a, float b, float c) {
    rot = {a, b, c};
    exp_rotate(a, b, c, p1, &p1);
    exp_rotate(a, b, c, p2, &p2);
    exp_rotate(a, b, c, p3, &p3);
    normal = {0,0,0};
    cross(p3 - p2, p3 - p1, &normal);
    normalize(&normal);
}

void Triangle::set_scale(float a, float b, float c) {
    sca = {a, b, c};
    scale(a, b, c, p1, &p1);
    scale(a, b, c, p2, &p2);
    scale(a, b, c, p3, &p3);
    normal = {0,0,0};
    cross(p3 - p2, p3 - p1, &normal);
    normalize(&normal);
}
*/


void Triangle::get_normal(valarray<float> p, valarray<float>* n) {
    world_to_obj(p, &p);
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
        bool is_ambient; 
        valarray<float> xyz;
        Color color;
        Light(valarray<float> p, Color c, bool is_d, bool);
        void light_vector(valarray<float> point, valarray<float> *l_vec);
};

Light::Light(valarray<float> p, Color c, bool is_d, bool is_a) {
    xyz = p;
    color = c;
    is_direct = is_d;
    is_ambient = is_a; 
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
    public :
        Shader();
        void phong(valarray<float> point, valarray<float> normal, valarray<float> view, Color * c, Material *mat);
};

Shader::Shader(void) {
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
        void reflectance_harshil(Ray, Color*, float);
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
    eye_pos = {0,0,-1.5};
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
    Film film;
    Raytracer raytracer;
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

Scene::Scene(valarray<float> eye, valarray<float> ll, valarray<float> lr, valarray<float> ul, valarray<float> ur, float rx, float ry) {
    eye_position = eye;
    LL = ll;
    LR = lr;
    UL = ul;
    UR = ur;
    sampler = Sampler();
    camera = Camera();
    raytracer = Raytracer();
    resolution_x = rx;
    resolution_y = ry;
    film = Film(resolution_x, resolution_y, 1);
}


bool loadOBJ(char *path, Material input_material);




