#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

//include header file for glfw library so that we can use OpenGL
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
static DWORD lastTime;
#else
static struct timeval lastTime;
#endif

#define PI 3.14159265 // Should be used from mathlib

using namespace std;
//****************************************************
// Simple init function
//****************************************************
void initializeRendering();

inline float sqr(float x) { return x*x; }

class Color {
    public:
        float red, green, blue;
        Color (float, float, float);
        Color(); 
        void set_color (float, float, float);
        void add_color(Color);
        };

Color::Color() {} 

Color::Color (float r, float g, float b) {
    red = r;
    green = g;
    blue = b;
}



class Light { 
  public:
    Color color;
    float x, y, z;
    bool direct;
    bool active = false;
    bool is_direct(void);
    bool is_active(void);
    Light(bool, float, float, float, float, float, float);
  };

bool Light::is_direct(void) { 
  return direct;
}

bool Light::is_active(void) { 
  return active;
}

Light::Light(bool is_dir, float r, float g, float b, float lx, float ly, float lz) {
  direct = is_dir;
  color = Color(r, g, b);
  x = lx;
  y = ly;
  z = lz;
}

void phong(float, float, float, Color *); 

void Color::set_color(float r, float g, float b) {
    red = r;
    green = g;
    blue = b;
}


// must make sure this stays in the bounds of color
void Color::add_color(Color c1) {
    red = c1.red + red;
    green  = c1.green + green;
    blue = c1.blue + blue;
}


class Vector {
    public:
        float x, y, z;
        Vector(float, float, float);
        void normalize(void);
        Vector();
        float get_length(void);
};
//*****************
//Vector Class and relevant functions
//****************

Vector::Vector() {} 

Vector::Vector (float v_x, float v_y, float v_z) {
    x = v_x;
    y = v_y;
    z = v_z;
}

void Vector::normalize (void) { 
    float length = sqrt(sqr(x) + sqr(y) + sqr(z));
    x = x/length;
    y = y/length;
    z = z/length; 
} 

float Vector::get_length(void) {
    return sqrt(sqr(x) + sqr(y) + sqr(z));
}


//*****************
//Point Class and relevant functions
//****************
// NB check if point should be struct or class
class Point { 
  public:
  float x, y, z;
  Point(float, float, float);
};

Point::Point (float p_x, float p_y, float p_z) {
    x = p_x;
    y = p_y;
    z = p_z;
}

//****************************************************
// A routine to set a pixel by drawing a GL point.  This is not a
// general purpose routine as it assumes a lot of stuff specific to
// this example.
//****************************************************
void setPixel(float x, float y, GLfloat r, GLfloat g, GLfloat b);

//****************************************************
// Keyboard inputs
//****************************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);


//****************************************************
// Draw a filled circle.
//****************************************************
void drawCircle(float centerX, float centerY, float radius);

//*****
//PHONG SHADING APPLIED
// px, py, pz : location of the pixel
// n : norm vector
// light_vecs : list of light vectors normalized
// light_cols : list of light colors 
// view : view vector
//*****
void phong(float px, float py, float pz, Color *pixel_color);
void ashikhim_shirley(float px, float py, float pz, Color *pixel_color);


float find_specular_power(Vector normal, Vector view, Vector light_vec);


void reflectance(Vector light_source, Vector normal, Vector *reflectance);

//****************************************************
// function that does the actual drawing of stuff
//***************************************************
void display( GLFWwindow* window );

//****************************************************
// function that is called when window is resized
//***************************************************
void size_callback(GLFWwindow* window, int width, int height);








