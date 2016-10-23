#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

//include header file for glfw library so that we can use OpenGL
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "shader.h"



#define PI 3.14159265 // Should be used from mathlib

using namespace std;

//****************************************************
// Global Variables
//****************************************************
GLfloat translation[3] = {0.0f, 0.0f, 0.0f};
int auto_strech = false;
int Width_global = 400;
int Height_global = 400;


//******
//HARD CODED INPUTS -- PLEASE CHANGE
//***
Color KA = Color(0.0, 0.0, 0.0);
Color KD = Color(0.0, 0.0, 0.0);
Color KS = Color(0.0, 0.0, 0.0);
float SPU = 1;
float SPV = 1;
bool is_isotropic = true;
bool ashikhmin_shirley = false;

int num_lights = 0;
int num_direct = 0;
int num_point = 0;

Light light1 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light2 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light3 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light4 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light5 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light6 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light7 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light8 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light9 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
Light light10 = Light(true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
vector<Light> lights = {light1,light2,light3,light4,light5,light6,light7,light8,light9,light10};

//****************************************************
// Simple init function
//****************************************************
void initializeRendering()
{
    glfwInit();
}
  
// must make sure this stays in the bounds of color

//****************************************************
// A routine to set a pixel by drawing a GL point.  This is not a
// general purpose routine as it assumes a lot of stuff specific to
// this example.
//****************************************************
void setPixel(float x, float y, GLfloat r, GLfloat g, GLfloat b) {
    glColor3f(r, g, b);
    glVertex2f(x+0.5, y+0.5);  // The 0.5 is to target pixel centers
    // Note: Need to check for gap bug on inst machines.
}

//Note: p1 is the head and p2 is the tail of the vector 
void points_to_vector(Point p1, Point p2, Vector *v) { 
  v->x = p1.x - p2.x; 
  v->y = p1.y - p2.y;
  v->z = p1.z - p2.z;
} 

void scale_vector(float c, Vector v, Vector *n) {
    n->x = c * v.x;
    n->y = c * v.y;
    n->z = c * v.z;
}

float dot(Vector v1, Vector v2) {
   return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

void add_vector(Vector v1, Vector v2, Vector *v) {
    v->x = v1.x + v2.x;
    v->y = v1.y + v2.y;
    v->z = v1.z + v2.z;
}

// must make sure this stays in the bounds of color
void scale_color(float c, Color c1, Color *c2) {
    c2->red = c * c1.red;
    c2->green = c * c1.green;
    c2->blue = c * c1.blue;
}

void mult_color(Color c1, Color c2, Color *c3) {
    c3->red = c1.red * c2.red;
    c3->green = c1.green * c2.green;
    c3->blue = c1.blue * c2.blue;
} 

void cross(Vector v1, Vector v2, Vector *v3)  {
    v3->x = v1.y * v2.z - v1.z * v2.y;
    v3->y = v1.z * v2.x - v1.x * v2.z;
    v3->z = v1.x * v2.y - v1.y * v2.x;
}

//****************************************************
// Keyboard inputs
//****************************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch (key) {
            
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_Q: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        case GLFW_KEY_LEFT :
            if (action) translation[0] -= 0.01f * Width_global; break;
        case GLFW_KEY_RIGHT:
            if (action) translation[0] += 0.01f * Width_global; break;
        case GLFW_KEY_UP   :
            if (action) translation[1] += 0.01f * Height_global; break;
        case GLFW_KEY_DOWN :
            if (action) translation[1] -= 0.01f * Height_global; break;
        case GLFW_KEY_F:
            if (action) auto_strech = !auto_strech; break;
        case GLFW_KEY_SPACE: break;
            
        default: break;
    }
    
}


//****************************************************
// Draw a filled circle.
//****************************************************
void drawCircle(float centerX, float centerY, float radius) {
    // Draw inner circle
    glBegin(GL_POINTS);

    // We could eliminate wasted work by only looping over the pixels
    // inside the sphere's radius.  But the example is more clear this
    // way.  In general drawing an object by loopig over the whole
    // screen is wasteful.

    int minI = max(0,(int)floor(centerX-radius));
    int maxI = min(Width_global-1,(int)ceil(centerX+radius));

    int minJ = max(0,(int)floor(centerY-radius));
    int maxJ = min(Height_global-1,(int)ceil(centerY+radius));

    for (int i = 0; i < Width_global; i++) {
        for (int j = 0; j < Height_global; j++) {

            // Location of the center of pixel relative to center of sphere
            float x = (i+0.5-centerX);
            float y = (j+0.5-centerY);

            float dist = sqrt(sqr(x) + sqr(y));

            if (dist <= radius) {

                // This is the front-facing Z coordinate
                float z = sqrt(radius*radius-dist*dist);
                Color pixel_color = Color();
                if (ashikhmin_shirley) { 
                  ashikhim_shirley(x, y, z, &pixel_color);
                } else { 
                  phong(x, y, z, &pixel_color);
                } 
                /*
                cout << "red" << pixel_color.red << endl;
                cout << "green" << pixel_color.green << endl;
                cout << "blue" << pixel_color.blue << endl;
                */
                // given x, y, z, and light_x, light_y, and light z:
                // find abient, diffuse, and specular parts
                // solve for normal, light, and viewing vectors
                // find refectance
                // solve for each part of phong
                // total phong
                
                setPixel(i, j, pixel_color.red, pixel_color.green, pixel_color.blue);

                // This is amusing, but it assumes negative Color values are treated reasonably.
                // setPixel(i,j, x/radius, y/radius, z/radius );
                
                // Just for fun, an example of making the boundary pixels yellow.
                // if (dist > (radius-1.0)) {
                //     setPixel(i, j, 1.0, 1.0, 0.0);
                // }
            }
        }
    }

    glEnd();
}

void fresnal(Vector half_angle, Vector view, Color *specular) { 
    Color ones = Color(1.0, 1.0, 1.0);
    Color negative_KS = Color();
    scale_color(-1.0, KS, &negative_KS);
    negative_KS.add_color(ones);
    float hv = 1.0 - dot(half_angle, view);
    float hv_coeff = pow(hv, 5); 
    Color tmp = Color();
    scale_color(hv_coeff, negative_KS, &tmp);
    tmp.add_color(KS);
    specular->red = tmp.red; 
    specular->green = tmp.green;
    specular->blue = tmp.blue;
    //cout << "FRESNAL : " << specular->red << "," << specular->green << "," << specular->blue << " half angle :" << half_angle.x << "," << half_angle.y << "," << half_angle.z << endl;
}

//*****
//ASHIKHIM SHIRLEY SHADING APPLIED
// px, py, pz : location of the pixel
// normal : norm vector
// view : view vector
// half_angle : normalized half vector between view and normal 
// parametric_u parametric_v : tangent vectors that form an orthonormal basis along with n 
// F(cos(theta)) : Fresnal reflectance for incident angle theta 
// light_vecs : list of light vectors normalized
// light_cols : list of light colors 
//*****


void ashikhim_shirley(float px, float py, float pz, Color *pixel_color) {
    Color tmp_pixel_color = Color(0.0, 0.0, 0.0);
    Vector view = Vector(0,0,1);
    Point cur_point = Point(px, py, pz);
    Vector normal = Vector(px, py, pz);
    normal.normalize();

    Color diffuse = Color(0.0, 0.0, 0.0);
    Color specular = Color(0.0, 0.0, 0.0);
    Color ambient = Color(0.0, 0.0, 0.0);

    for(int d =0; d < num_lights; d++) {
      Light cur_light = lights[d];
      Vector light_vec = Vector();
      Color light_col = cur_light.color;

      if(cur_light.is_direct()) {
        light_vec = Vector(-1 * cur_light.x, -1 * cur_light.y, -1 * cur_light.z);
        light_vec.normalize();
      } else {
        Point cur_light_pt = Point(cur_light.x, cur_light.y, cur_light.z);
        points_to_vector(cur_light_pt, cur_point, &light_vec);
        light_vec.normalize();
      }

      Vector half_angle = Vector(0.0, 0.0, 0.0); 
      add_vector(light_vec, view, &half_angle); 
      half_angle.normalize();

      //Hypothetical Ambient 
      /*
      Color ambient = Color(0.0, 0.0, 0.0);
      Vector reflect = Vector();
      reflectance(light_vec, normal, &reflect);
      reflect.normalize();
      Color new_ambient = Color();
      mult_color(KA, light_col, &new_ambient);
      //cout << "Ambient Parts -- KA: " << KA.red << ", " << KA.green << ", " << KA.blue << endl;
      ambient.add_color(new_ambient);
      */

      //Calculate specular
      Color new_specular = Color(0.0, 0.0, 0.0);
      float p = find_specular_power(normal, view, light_vec);
      //numerator = sqrt((p_u+1)(p_v+1))*dot(n, half_angle)^specpower
      float numerator = sqrt((SPU + 1) * (SPV + 1)) * pow (dot(normal, half_angle), p);
      //denominator = 8*pi*dot(h, view)*max(dot(n, view), dot(n, light) 
      float tmp_he = dot(half_angle, view); 
      float tmp_ne = dot(normal, view); 
      float tmp_nl = dot(normal, light_vec); 
      float mx = max(tmp_ne, tmp_nl);
      float denominator = 8*PI*tmp_he*mx;
      //float denominator = 8*PI*tmp_he*max(dot(normal, view), dot(normal, light_vec));
      float new_specular_coeff = numerator/denominator; 
      Color tmp = Color();
      fresnal(half_angle, view, &tmp);
      scale_color(new_specular_coeff, tmp, &new_specular);
      specular.add_color(new_specular);
      
      //Calculate diffuse 
      Color new_diffuse = Color();
      float half_nv = 1 - dot(normal, view)/2; 
      float half_nl = 1 - dot(normal, light_vec)/2; 
      float diffuse_coeff = (1 - pow(half_nv, 5))*(1 - pow(half_nl, 5));
      diffuse_coeff = (28/(23*PI))*diffuse_coeff;
      Color diff1 = Color();
      Color ones = Color(1.0, 1.0, 1.0);
      Color negative_KS = Color();
      scale_color(-1.0, KS, &negative_KS);
      negative_KS.add_color(ones);
      mult_color(negative_KS, KD, &diff1);
      scale_color(diffuse_coeff, diff1, &new_diffuse);
      diffuse.add_color(new_diffuse);
      /*
      cout << "NORMAL: " << normal.x << "," << normal.y << "," << normal.z << endl;
      cout << "LIGHT VEC " << light_vec.x << "," << light_vec.y << "," << light_vec.z << endl;
    
      cout << "Diffuse: half_nl, "  << half_nl<< ", half_nv "<< half_nv << ", diffuse coeff:" << diffuse_coeff << endl;
      cout << "KS: " << KS.red << "," << KS.green << "," << KS.blue <<  ", Negative KS: " << negative_KS.red << ","<< negative_KS.green << "," << negative_KS.blue << "  KD: " << KD.red << ","<< KD.green << KD.blue << endl;
      cout << "NEW DIFFUSE: " << new_diffuse.red << endl;
      */
    }
  tmp_pixel_color.add_color(diffuse); 
  tmp_pixel_color.add_color(specular); 
  /*
  tmp_pixel_color.add_color(ambient); 
  */
  pixel_color->red = tmp_pixel_color.red;
  pixel_color->green = tmp_pixel_color.green;
  pixel_color->blue = tmp_pixel_color.blue;

       /*
      cout << "spec VAL r: " << specular.red << "; g: " << specular.green << "; b: " << specular.blue << endl;
      cout << "diffuse VAL r: " << diffuse.red << "; g: " << diffuse.green << "; b: " << diffuse.blue << endl;
      cout << "ambient VAL r: " << ambient.red << "; g: " << ambient.green << "; b: " << ambient.blue << endl;
      */
      
}


//*****
//PHONG SHADING APPLIED
// px, py, pz : location of the pixel
// n : norm vector
// light_vecs : list of light vectors normalized
// light_cols : list of light colors 
// view : view vector
//*****
void phong(float px, float py, float pz, Color *pixel_color) {
    Color tmp_pixel_color = Color(0.0, 0.0, 0.0);
    Vector view = Vector(0,0,1);
    Point cur_point = Point(px, py, pz);
    Vector normal = Vector(px, py, pz);
    normal.normalize();
    /*
    cout << "Normal Vector: " << normal.x << ", " << normal.y << ", " << normal.z << endl;
    */
    Color ambient = Color(0.0, 0.0, 0.0);
    Color diffuse = Color(0.0, 0.0, 0.0);
    Color specular = Color(0.0, 0.0, 0.0);

    //Should combine these two for loops into helper method

    for(int d =0; d < num_lights; d++) {
      //cout << "num_lights: " << d << endl;
      Light cur_light = lights[d];

      //cout << "light point: x - " << cur_light.x << " y - " << cur_light.y << " z - " << cur_light.z << "light red: " << cur_light.color.red << "light green: " << cur_light.color.green << "light blue: " << cur_light.color.blue << endl;

      Vector light_vec = Vector();
      Color light_col = cur_light.color;
      //cout << "r: " << light_col.red << endl;
      //cout << "g: " << light_col.green << endl;
      //cout << "b: " << light_col.blue << endl;

      if(cur_light.is_direct()) {
        light_vec = Vector(-1 * cur_light.x, -1 * cur_light.y, -1 * cur_light.z);
        light_vec.normalize();
      } else {
      //cout << "in loop" << endl;
      //cout << "light point: x - " << cur_light.x << " y - " << cur_light.y << " z - " << cur_light.z << "light red: " << cur_light.color.red << "light green: " << cur_light.color.green << "light blue: " << cur_light.color.blue << endl;

        Point cur_light_pt = Point(cur_light.x, cur_light.y, cur_light.z);
        //cout << "current point: x - " << px << " y - " << py << " z - " << pz << endl;
        //cout << "light point: x - " << cur_light_pt.x << " y - " << cur_light_pt.y << " z - " << cur_light_pt.z << endl;
        points_to_vector(cur_light_pt, cur_point, &light_vec);
        light_vec.normalize();
        //cout << "current light vector: x - " << light_vec.x << " y: " << light_vec.y << " z: " << light_vec.z << endl;
      }

      Vector reflect = Vector();
      reflectance(light_vec, normal, &reflect);
      reflect.normalize();
      Color new_ambient = Color();
      mult_color(KA, light_col, &new_ambient);
      //cout << "Ambient Parts -- KA: " << KA.red << ", " << KA.green << ", " << KA.blue << endl;
      ambient.add_color(new_ambient);

      Color new_diffuse = Color();
      Color diff1 = Color();
      float l_n = dot(light_vec, normal);
      //cout << "l dot n: " << l_n << ", normal Vector: " << normal.x << ", " << normal.y << ", " << normal.z <<", current light vector: x - " << light_vec.x << " y: " << light_vec.y << " z: " << light_vec.z << endl;
      float positive_dot = max(l_n,(float)  0.0);
      mult_color(KD, light_col, &diff1);
      /*
      cout << "KD * light col: " << diff1.red << "," << diff1.green << "," << diff1.blue << endl;
      */
      scale_color(positive_dot, diff1, &new_diffuse);
      diffuse.add_color(new_diffuse);
      
      
      Color new_specular = Color();
      Color spec1 = Color();
      float ref_view = dot(reflect, view);
      float mx = max(ref_view, (float) 0.0);
      float power = find_specular_power(normal, view, light_vec);
      float tmp = pow(mx, power);
      scale_color(tmp, KS, &spec1);
        /*
        cout << "KS: " << KS.red << "," << KS.green << "," << KS.blue << endl;
        cout << "max: " << mx << ", ref dot view" << ref_view << "," << spec1.blue << endl;
        cout << "KS * power: " << spec1.red << "," << spec1.green << "," << spec1.blue << endl;
        */
      mult_color(spec1, light_col, &new_specular);
      /* 
      cout << "(KS * power) * light color: " << new_specular.red << "," << new_specular.green << "," << new_specular.blue << endl;
      cout << "Specular Parts -- pow:" << tmp << "; REFLECT:" << reflect.x << "," << reflect.y << "," << reflect.z<< " DOT: "<< ref_view << endl;
       */ 
      specular.add_color(new_specular);
      /* 
      cout << "spec VAL r: " << specular.red << "; g: " << specular.green << "; b: " << specular.blue << endl;
      cout << "diffuse VAL r: " << diffuse.red << "; g: " << diffuse.green << "; b: " << diffuse.blue << endl;
      cout << "ambient VAL r: " << ambient.red << "; g: " << ambient.green << "; b: " << ambient.blue << endl;
      */
    }

  tmp_pixel_color.add_color(ambient); 
  tmp_pixel_color.add_color(diffuse); 
  tmp_pixel_color.add_color(specular); 
  pixel_color->red = tmp_pixel_color.red;
  pixel_color->green = tmp_pixel_color.green;
  pixel_color->blue = tmp_pixel_color.blue;
/*
  cout << "spec VAL r: " << specular.red << "; g: " << specular.green << "; b: " << specular.blue << endl;
  cout << "diffuse VAL r: " << diffuse.red << "; g: " << diffuse.green << "; b: " << diffuse.blue << endl;
  cout << "ambient VAL r: " << ambient.red << "; g: " << ambient.green << "; b: " << ambient.blue << endl;
  cout << "FINAL PIXEL VAL r: " << tmp_pixel_color.red << "; g: " << tmp_pixel_color.green << "; b: " << tmp_pixel_color.blue << endl;
*/

}

float find_specular_power(Vector normal, Vector view, Vector light_vec) {
    float p;
    if(is_isotropic) {
        p = SPU;
    } else {
        //find half angle h = norm(l + v) 
        Vector half_angle = Vector(0.0, 0.0, 0.0); 
        add_vector(light_vec, view, &half_angle); 
        half_angle.normalize();

        //find v vector = norm(y - n(n*y)) y = [0,1,0]
        Vector parametric_v = Vector(0.0, 0.0, 0.0);
        Vector y = Vector(0.0, 1.0, 0.0);
        float tmp = -1*dot(normal, y); 
        Vector scaled_normal = Vector(0.0, 0.0, 0.0); 
        scale_vector(tmp, normal, &scaled_normal);
        add_vector(y, scaled_normal, &parametric_v);
        parametric_v.normalize();

        //find u vector = norm(cross(v, n)) 
        Vector parametric_u = Vector(0.0, 0.0, 0.0); 
        cross(parametric_v, normal, &parametric_u); 
        parametric_u.normalize();

        //find specular exponent = p_u(h*u)^2 + p_v(h*v)^2/1 - (h*n)^2
        float tmp_hu = SPU*pow(dot(half_angle, parametric_u), 2); 
        float tmp_hv = SPV*pow(dot(half_angle, parametric_v), 2); 
        float denominator = 1 - pow(dot(half_angle, normal), 2);
        p = (tmp_hu + tmp_hv)/denominator;
    } 
    return p;
}

void find_specular_color(Vector norm, Vector light_vec, Vector view, Color *total_specular) {
    float new_red = 0.0;
    float new_green = 0.0;
    float new_blue = 0.0;

    total_specular->red = new_red;
    total_specular->green = new_green;
    total_specular->blue = new_blue;
}

void reflectance(Vector light_source, Vector normal, Vector *reflectance) { 
    Vector negative_norm = Vector();
    scale_vector(-1, light_source, &negative_norm); 
    float tmp = 2.0*dot(light_source, normal); 
    Vector scaled_norm = Vector();
    scale_vector(tmp, normal, &scaled_norm); 
    add_vector(negative_norm, scaled_norm, reflectance);
} 


//****************************************************
// function that does the actual drawing of stuff
//***************************************************
void display( GLFWwindow* window )
{
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f ); //clear background screen to black
    
    glClear(GL_COLOR_BUFFER_BIT);                // clear the Color buffer (sets everything to black)
    
    glMatrixMode(GL_MODELVIEW);                  // indicate we are specifying camera transformations
    glLoadIdentity();                            // make sure transformation is "zero'd"
    
    //----------------------- code to draw objects --------------------------
    glPushMatrix();
    glTranslatef (translation[0], translation[1], translation[2]);
    drawCircle(Width_global / 2.0 , Height_global / 2.0 , min(Width_global, Height_global) * 0.8 / 2.0);
    glPopMatrix();
    
    glfwSwapBuffers(window);
    
}

//****************************************************
// function that is called when window is resized
//***************************************************
void size_callback(GLFWwindow* window, int width, int height)
{
    // Get the pixel coordinate of the window
    // it returns the size, in pixels, of the framebuffer of the specified window
    glfwGetFramebufferSize(window, &Width_global, &Height_global);
    
    glViewport(0, 0, Width_global, Height_global);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, Width_global, 0, Height_global, 1, -1);
    
    display(window);
}

void create_ambient_light(char* red, char* green, char* blue) {
    float r =(float) atof(red);
    float g =(float) atof(green);
    float b =(float) atof(blue);
    KA = Color(r, g, b);
}

void create_diffuse_light(char* red, char* green, char* blue) {
    float r =(float) atof(red);
    float g =(float) atof(green);
    float b =(float) atof(blue);
    KD = Color(r, g, b);
}

void create_specular_light(char* red, char* green, char* blue) {
    float r =(float) atof(red);
    float g =(float) atof(green);
    float b =(float) atof(blue);
    KS = Color(r, g, b);
}

void create_light(char* x1, char* y1, char* z1, char* red, char* green, char* blue, bool is_direct) {
    int idx = num_lights;
    float radius = min(Width_global, Height_global) * 0.8;
    /*
    cout << "RADIUS" << radius << endl;
    */
    float x =(float) atof(x1) * radius;
    float y =(float) atof(y1) * radius;
    float z =(float) atof(z1) * radius;
    float r =(float) atof(red);
    float g =(float) atof(green);
    float b =(float) atof(blue);
    lights[idx].color.red = r;
    lights[idx].color.green = g;
    lights[idx].color.blue = b;
    lights[idx].x = x;
    lights[idx].y = y;
    lights[idx].z = z;
    lights[idx].active = true;
    lights[idx].direct = is_direct;
    num_lights++;
}


//****************************************************
// the usual stuff, nothing exciting here
//****************************************************
int main(int argc, char *argv[]) {
    //This initializes glfw
    initializeRendering();
    int i = 0;
    //cout << "arg c: " << argc << endl;
    while( i + 1 <= argc ) {
        //Maybe try make this an enum at the end 
        //cout << "1" << endl;
        //cout << "accessing new arg" << i << endl;
        //cout << argv[i] << endl;
        //cout << "iteration: " << i << endl;
        if (strcmp(argv[i], "-ka") == 0) {
            //cout << "in ka" << endl;
            create_ambient_light(argv[i+1], argv[i+2], argv[i+3]);
            i = i + 3;
            continue;
        } else if (strcmp(argv[i], "-kd") == 0) {
            //cout << "in kd" << endl;
            create_diffuse_light(argv[i+1], argv[i+2], argv[i+3]);
            //cout << KD.red << endl;
            //cout << KD.green << endl;
            //cout << KD.blue << endl;
            i = i + 3;
        } else if (strcmp(argv[i], "-ks") == 0) {
            //cout << "in ks" << endl;
            create_specular_light(argv[i+1], argv[i+2], argv[i+3]);
            //cout << KS.red << endl;
            //cout << KS.green << endl;
            //cout << KS.blue << endl;
            i = i + 3;
        } else if (strcmp(argv[i], "-spu") == 0) { 
            //cout << "in spu" << endl;
            SPU = (float) atof(argv[i+1]);
            is_isotropic = false;
            i++;
        } else if (strcmp(argv[i], "-spv") == 0) { 
            //cout << "in spv" << endl;
            SPV = (float) atof(argv[i+1]);
            is_isotropic = false;
            i++;
        } else if (strcmp(argv[i], "-sp") == 0) { 
            //cout << "in sp" << endl;
            SPU = (float) atof(argv[i+1]);
            SPV = (float) atof(argv[i+1]);
            is_isotropic = true;
            i++;
        }  else if (strcmp(argv[i], "-pl") == 0) { 
            create_light(argv[i+1],argv[i+2],argv[i+3],argv[i+4],argv[i+5],argv[i+6], false);
            i = i + 6;
            num_point++;
        }  else if (strcmp(argv[i], "-dl") == 0) { 
            create_light(argv[i+1],argv[i+2],argv[i+3],argv[i+4],argv[i+5],argv[i+6], true);
            i = i + 6;
            num_direct++;
        } else if (strcmp(argv[i], "-asm") == 0) { 
            /*
            cout << "here" << endl;
            */
            ashikhmin_shirley = true; 
        }
        i = i + 1;
    }
    //cout << "2" << endl;
    /*
    cout << "ALL PARSED INFO DONE ------ " << endl;
    for(int j = 0; j < num_lights; j++) {
        cout << "light point: x - " << lights[j].x << " y - " << lights[j].y << " z - " << lights[j].z << "light red: " << lights[j].color.red << "light green: " << lights[j].color.green << "light blue: " << lights[j].color.blue << endl;
    }

    cout << "KD r:" << KD.red << endl;
    cout << "KD g:" << KD.green << endl;
    cout << "KD b:" << KD.blue << endl;

    cout << "KS r:" << KS.red << endl;
    cout << "KS g:" << KS.green << endl;
    cout << "KS b: "<< KS.blue << endl;

    cout << "KA r: " << KA.red << endl;
    cout << "KA g: "<< KA.green << endl;
    cout << "KA b: " << KA.blue << endl;

    cout << "SPU: " << SPU << "; SPV: " << SPV << endl;

    cout << " ------ " << endl;
    */
    GLFWwindow* window = glfwCreateWindow( Width_global, Height_global, "CS184", NULL, NULL );
    if ( !window )
    {
        cerr << "Error on window creating" << endl;
        glfwTerminate();
        return -1;
    }
    
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    if ( !mode )
    {
        cerr << "Error on getting monitor" << endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent( window );
    
    // Get the pixel coordinate of the window
    // it returns the size, in pixels, of the framebuffer of the specified window
    glfwGetFramebufferSize(window, &Width_global, &Height_global);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, Width_global, 0, Height_global, 1, -1);
    
    glfwSetWindowTitle(window, "CS184");
    glfwSetWindowSizeCallback(window, size_callback);
    glfwSetKeyCallback(window, key_callback);
    
    while( !glfwWindowShouldClose( window ) ) // infinite loop to draw object again and again
    {   // because once object is draw then window is terminated
        display( window );
        
        if (auto_strech){
            glfwSetWindowSize(window, mode->width, mode->height);
            glfwSetWindowPos(window, 0, 0);
        }
        
        glfwPollEvents();
        
    }

    return 0;
}








