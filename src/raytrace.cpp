#include "raytrace.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
Scene *scn;
bool camera_set = false; 
bool obj_parsed = false;
int x_resolution = 200;
int y_resolution = 200;
int samples_per_pix = 0;
float max_recursive_depth = 0;
float reflectivity = 0.75;
Color sky = Color(70.0/255.0, 130.0/255.0, 180/255.0);
/*
Color good_ambient = Color( 119.0/255.0,  139.0/255.0, 165.0/255.0);
valarray<float> c1 = {0.0, 0.0, 10.0};
Color KA = Color(0.5, 0.5, 0.5);
//Color KA = Color(0.0, 0.0, 0.0);
Color KD = Color(0.2, 0.3, 0.8);
//Color KD = Color(0.0, 0.0, 0.0);
Color KS = Color(0.2, 0.3, 0.8);
//Color KS = Color(0.0, 0.0, 0.0);
float SPU = 2;
float SPV = 2;
valarray<float> c2 = {20.0, 0.0, 50.0};
Color KA1 = Color(0.2, 0.2, 0.2);
//Color KA = Color(0.0, 0.0, 0.0);
Color KD1 = Color(0.9, 0.5, 0.1);
//Color KD = Color(0.0, 0.0, 0.0);
Color KS1 = Color(0.8, 0.4, 0.0);
float SPU1 = 2;
float SPV1 = 2;
Material m1 = Material(good_ambient, KD, KS, SPU, SPV);
Material m2 = Material(KA1, KD1, KS1, SPU1, SPV1);
Sphere* s1 = new Sphere(c1, 4.0, m1); 
Sphere* s2 = new Sphere(c2, 1.0, m2); 
//Object* objects[] = {s1,s2};
valarray<float> p1 = {-1, -1, 10};
valarray<float> p2 = {0, 1, 10};
valarray<float> p3 = {1, -1, 10};
Triangle* t1 = new Triangle(p1, p2, p3, m1);
Func_Sphere* fs1 = new Func_Sphere(c1, 1.0, m1);
Func_Sphere* fs2 = new Func_Sphere(c1, 1.0, m1);
vector<Object*> objects = {fs1};
*/
Color ka = Color(0.4, 0.4, 0.4);
Color kd = Color(0.3, 0.3, 0.0);
Color ks = Color(0.3, 0.3, 0.0);
Color kr = Color(0.0, 0.0, 0.0);
float SPU = 2;
float SPV = 2;
Material default_material = Material(ka, kd, ks, kr, SPU, SPV);
vector<Object*> objects = {};
vector<Light*> lights = {};
void dist(valarray<float> p1, valarray<float> p2, float* d);
/*
 *                      FILM CLASS
 * The film class holds our data structure where we store our image.
 * It's main functions are commit and write image. Commit assigns a 
 * pixel a color by pushing it back to a vector of existing colors. 
 * When we write the image we average over each pixel bucket and use 
 * an opensource header file to save the image.
 */

Film::Film(int x_res, int y_res, int sr) {
    res_x = x_res;
    res_y = y_res;
    sample_rate = sr;
    pixel_buckets = vector<vector<vector<Color>>> (x_res, vector<vector<Color>>(y_res, vector<Color>(sr, Color(0,0,0))));
}

Film::Film(void) {
    res_x = x_resolution;
    res_y = y_resolution;
    sample_rate = samples_per_pix;;
    pixel_buckets = vector<vector<vector<Color>>> (x_resolution, vector<vector<Color>>(y_resolution, vector<Color>(samples_per_pix, Color(0,0,0))));
}

// Averages over each pixel bucks and writes the result to a screen
void Film::write_image(void) {
    cout << "ATTEMPTING TO WRITE IMAGE" << endl;
    char const* filename = "test_image";
    // DO NOT LEAVE THIS. THIS IS A TEST. 
    int stride = 3*res_x;
    int comp = 3;
    int cur_index;
    uint8_t avg_pixels[res_x * res_y * 3];
    // NEEDS TO BE REPLACED WITH FINDING AVERGE FOR EACH PIXEL BUCKET
    for(int w_row = 0; w_row < res_y; w_row++) {
        for(int col = 0; col < res_x; col++) {
            int row = res_y - w_row - 1;
            //int col = res_x - w_col - 1; // DOUBLE CHECK THESE PLEASE
            int p_index = row*res_x + col;
            cur_index = 3*p_index;
            Color cur_color = pixel_buckets[row][col][0];
            if (cur_color.r > 1) {
                cout << "red bigger" << endl;
                cout << cur_color.r << endl;
                cur_color.r = 1.0;
            } 
            if (cur_color.g > 1) {
                cout << "green bigger" << endl;
                cout << cur_color.g << endl;
                cur_color.g = 1.0;
            } 
            if (cur_color.b > 1) {
                cout << "blue bigger" << endl;
                cout << cur_color.b << endl;
                cur_color.b = 1.0;
            } 
            avg_pixels[cur_index] = cur_color.r * 255;
            avg_pixels[cur_index+1] = cur_color.g * 255;
            avg_pixels[cur_index+2] = cur_color.b * 255;
        }
    }

    stbi_write_png(filename, res_x, res_y, comp, &avg_pixels, stride);
    cout << "FILE WRITTEN" << endl;
}

// Add color c to the bucket of colors for sample s
void Film::commit(Sample s, Color c) {
    //cout << "Committing color to film "<< s.x << ", " << s.y << endl;
    int col = s.x;
    int row = s.y;
    // THIS IS AN IMPLEMENTATION WITH ONLY ONE SAMPLE PER BUCKET
    // NO ANTI ALIASING
    pixel_buckets[row][col][0] = c;
    //cout << "Commit Sucess" << endl;

}


/*
 *                  SAMPLER CLASS
 * For now, the sampler just generates a single sample in the center of each pixel. 
 * It will generate (x,y) of a screen sample and return true. Next time it gets called, 
 * it will generate another sample for the next pixel. It will return false when all the
 * samples from all the pixels are generated. In our case, we generate 1 sample per pixel, 
 * at the pixel sample. Later on, if we want to do multi-sample per pixel, we need to 
 * modify this class.
 *
 */
Sampler::Sampler(void) {
    current_px = 0;
    current_py = 0;
    max_x = x_resolution - 1;
    max_y = y_resolution - 1;
}

Sampler::Sampler(int x_res, int y_res) {
    current_px = 0;
    current_py = 0;
    max_x = x_res-1;
    max_y = y_res-1;
}


bool Sampler::get_sample(Sample *sample){
    // set this sample to the current sample (in screen coords) 
    sample->x = current_px;
    sample->y = current_py;
    // update next sample
    if (current_px < max_x) {
        current_px += 1;
    } else if (current_py < max_y) {
        current_px = 0;
        current_py += 1;
    } else {
        return false;
    }
    return true;
}
/*
 *                      OBJECT CLASS
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

bool Sphere::t_hit(Ray ray, float* t) {
    //cout<< "in sphere t_hit" << endl;
    valarray<float> d = ray.direction;
    world_to_obj(d, &d);
    valarray<float> e = ray.point;
    world_to_obj(e, &e);
    float discriminant = pow(dot(d,(e - center)), 2) - (dot(d, d)) * (dot(e - center, e - center) - pow(radius, 2));
    if(discriminant < 0.0) {
        *t = -1;
        return false;
    }
    float numerator = -1.0 * (dot(d, (e-center)));
    float denominator = dot(d,d);
    if (discriminant > 0) {
        float t1 = (numerator + discriminant) / denominator;
        float t2 = (numerator - discriminant) / denominator;
        *t = min(t1, t2);
        return true;
    } else {
        *t = numerator / denominator;
        return true;
        }
}

// With points in counterclockwise direction, ONE, TWO and THREE,
// returns if a ray intersects a triangle. If there is a hit, calculates 
// the corresponding ray's t value at the intersection, as well as 
// BETA and GAMMA satisfying:
// RAY(t) = ONE + BETA * TWO + GAMMA * THREE
bool Triangle::t_hit(Ray ray, float *t) {
    valarray<float> r1 = ray.point;
    world_to_obj(r1, &r1);
    valarray<float> r2 = ray.direction;
    world_to_obj(r2, &r2); 
    float a = p1[0] - p2[0];
    float b = p1[1] - p2[1];
    float c = p1[2] - p2[2];

    float d = p1[0] - p3[0];
    float e = p1[1] - p3[1];
    float f = p1[2] - p3[2];

    float g = r2[0];
    float h = r2[1];
    float i = r2[2];

    float j = p1[0] - r2[0];
    float k = p1[1] - r2[1];
    float l = p1[2] - r2[2];

    float ei = e*i;
    float hf = h*f;
    float gf = g*f;
    float di = d*i;
    float dh = d*h;
    float eg = e*g;

    float M = a * (ei - hf) + b * (gf - di) + c * (dh - eg);
    float beta = (j*(ei - hf) + k*(gf - di) + l*(dh - eg)) / M ;
    float gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c)) / M;
    float t_found = (-1) * (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c)) / M;
    
    if (t_found < 0 || beta < 0 || gamma < 0 || (beta + gamma) > 1) {
        *t = -1;
        return false;
    }

    *t = t_found;
    return true;
}

/*
 *                      SHADER CLASS
 * This holds the light environment of our scene. Lights is a 
 * list of lights that is currently in our environment. We can 
 * call phong on an object, normal, point and view to be able 
 * to calculate the resulting color. 
 *
 */

bool shadow_hit(Light light, valarray<float> point) {
    float epsilon = 0.1;
    float max_t = 1000;
    valarray<float> light_dir = {0.0,0.0,0.0};
    light.light_vector(point, &light_dir);
    Ray s = Ray(point, light_dir);
    // ONLY WORKS FOR POINT LIGHT
    valarray<float> l_p = -1 * light_dir;
    Ray light_ray = Ray(light.xyz, l_p);
    if(light.is_direct) {
        valarray<float> inf_pt = {0.0,0.0,0.0};
        s.eval(max_t, &inf_pt);
        light_ray = Ray(inf_pt, l_p);
    }
    float t = 0.0;
    bool light_hit = true;
    float light_t = sqrt(pow((light_ray.point-point), 2).sum());
    for (Object* obj : objects) {
        if(obj->t_hit(light_ray, &t)) {
            if (abs(t - light_t) < epsilon) {
                valarray<float> cord = {0.0,0.0,0.0};;
                light_ray.eval(t, &cord);
                float norm = sqrt(pow((cord-point), 2).sum());
                if (norm > epsilon) {
                    light_hit = false;
                    //cout << "shadow ray hit" << endl;
                }
            }
        }
            
    }
    return light_hit;
}


void Shader::phong(valarray<float> point, valarray<float> normal, valarray<float> view, Color *c, Material *obj) {
    Color tmp_pixel_color = Color(0.0, 0.0, 0.0);
    valarray<float> cur_point = point;
    Color ambient = Color(0.0, 0.0, 0.0);
    Color diffuse = Color(0.0, 0.0, 0.0);
    Color specular = Color(0.0, 0.0, 0.0);
    if (false) {
    cout << " CURRENTLY PROCESSING POINT : " << point[0] << "," << point[1] << "," << point[2] << endl;

    cout << "normal vector " << normal[0] << " " << normal[1] << " " << normal[2] << endl;
    cout << "view vector " << view[0] << " " << view[1] << " " << view[2] << endl;
    }
    cout << "light vector size " << lights.size() << endl;
    for(int d = 0; d < lights.size(); d++) {
      Light cur_light = *lights[d];
      valarray<float> light_vec = {0.0,0.0,0.0};
      cur_light.light_vector(point, &light_vec);
      if (true) {
      cout << "light vector " << light_vec[0] << " " << light_vec[1] << " " << light_vec[2] << endl;
      }
      Color light_col = cur_light.color;
      valarray<float> reflect = {0.0,0.0,0.0};
      reflectance(light_vec, normal, &reflect);
      if (false) {
      cout << "reflect vector " << reflect[0] << " " << reflect[1] << " " << reflect[2] << endl;
      }
      //AMBIENT
      Color new_ambient = Color();
      mult_color(obj->KA, light_col, &new_ambient);
      ambient.add_color(new_ambient);
      if (cur_light.is_ambient) { 
          continue; 
      } 
      else if (shadow_hit(cur_light, point)) {
      //if (false) {
          //cout<< "RETURNED TRUE" << endl;
          //DIFFUSE
          Color new_diffuse = Color();
          Color diff1 = Color();
          float l_n = dot(light_vec, normal);
          float positive_dot = max(l_n,(float)  0.0);
          mult_color(obj->KD, light_col, &diff1);
          scale_color(positive_dot, diff1, &new_diffuse);
          diffuse.add_color(new_diffuse);
          if (false) {
          cout << "l dot n : " << l_n << endl;
          cout << "after mult by KD " << diff1.r << diff1.g << diff1.b << endl;
          cout << "after scale " << new_diffuse.r << new_diffuse.g << new_diffuse.b << endl;
          }
          //SPECULAR 
          Color new_specular = Color();
          Color spec1 = Color();
          float ref_view = dot(reflect, view);
          if (false) {
          cout << "dot of reflect and view: " << ref_view << endl;
          }
          float mx = max(ref_view, (float) 0.0);
          float power = find_specular_power(normal, view, light_vec, obj);
          float tmp = pow(mx, power);
          scale_color(tmp, obj->KS, &spec1);
          mult_color(spec1, light_col, &new_specular);
          specular.add_color(new_specular);

      }
    }
  tmp_pixel_color.add_color(ambient); 
  tmp_pixel_color.add_color(diffuse); 
  if (false) {
  cout << "diffuse r,g,b: " << diffuse.r << " " << diffuse.g << " " << diffuse.b << endl;
  }
  tmp_pixel_color.add_color(specular); 
  if (false) {
  cout << "specular r,g,b: " << specular.r << " " << specular.g << " " << specular.b << endl;
  }

  c->r = min(tmp_pixel_color.r, (float) 1.0);
  c->g = min(tmp_pixel_color.g, (float) 1.0);
  c->b = min(tmp_pixel_color.b, (float) 1.0);
  if (c->r + c->g + c->b <= 0.01) {
    c->r = sky.r;
    c->g = sky.g;
    c->b = sky.b;
  }
}

void reflectance(valarray<float> light_source, valarray<float> normal, valarray<float> *reflectance) { 
    *reflectance = -1*light_source + 2*dot(light_source, normal)*normal;
    valarray<float> r = *reflectance;
    //cout << "reflect vector pre-normalize " << r[0] << " " << r[1] << " " << r[2] << endl;
    normalize(reflectance);
    normalize(&r);
    //cout << "r vector post-normalize " << r[0] << " " << r[1] << " " << r[2] << endl;
}

float find_specular_power(valarray<float> normal, valarray<float> view, valarray<float> light_vec, Material* obj) {
    float p;
    if(obj->SPU == obj->SPV) {
        p = obj->SPU;
    } else {
        //find half angle h = norm(l + v) 
        valarray<float> half_angle = light_vec +view;
        normalize(&half_angle);

        //find v vector = norm(y - n(n*y)) y = [0,1,0]
        valarray<float> parametric_v = {0.0, 0.0, 0.0};
        valarray<float> y = {0.0, 1.0, 0.0};
        float tmp = -1*dot(normal, y); 
        valarray<float> scaled_normal = tmp * normal;
        parametric_v = y + scaled_normal;
        normalize(&parametric_v);

        //find u vector = norm(cross(v, n)) 
        valarray<float> parametric_u = {0.0, 0.0, 0.0}; 
        cross(parametric_v, normal, &parametric_u); 
        normalize(&parametric_u);

        //find specular exponent = p_u(h*u)^2 + p_v(h*v)^2/1 - (h*n)^2
        float tmp_hu =(obj->SPU)*pow(dot(half_angle, parametric_u), 2); 
        float tmp_hv = (obj->SPV)*pow(dot(half_angle, parametric_v), 2); 
        float denominator = 1 - pow(dot(half_angle, normal), 2);
        p = (tmp_hu + tmp_hv)/denominator;
    } 
    return p;
}


/*
 *                      RAYTRACER CLASS
 * Raytracer traces a ray - this is its main function. The raytracer takes
 * a ray and knows a shading a environment. It sends the ray out through our
 * view window and sees if it intersects with any object if it does interesect 
 * it uses the shader to compute the resulting shader value. 
 *
 */

void Raytracer::trace(Ray r, Color *c) {
    //cout << "RAY TRACER TRACING" << endl;
    valarray<float> view = -1 * r.direction;
    normalize(&view);
    Color next_color = Color();
    // for all obj polynomials, see if we hit, order them, figure out who is hit first
    for(Object* cur_object : objects) {
        float t = 0.0;
        //cout << "Trying to hit" << endl;
        if (cur_object->t_hit(r, &t)) {
            //cout << "HIT!" << endl;
            valarray<float> cord = {0,0,0};
            // NOW CORD IS OUR x,y,z point of intersection. 
            r.eval(t, &cord);
            valarray<float> cur_norm = {0.0,0.0,0.0};
            cur_object->get_normal(cord, &cur_norm);
            shader.phong(cord, cur_norm, view, c, &cur_object->material);
            if (c->r + c->g + c->b != 0) {
                valarray<float> reflect = {0.0,0.0,0.0};
                reflectance(-1 * r.direction , cur_norm, &reflect);
                Ray new_ray = Ray(cord, reflect);
                reflectance_harshil(new_ray, &next_color, 0);
            }
            c->add_color(next_color);
        }
    }
}

void Raytracer::reflectance_harshil(Ray ray, Color *c, float depth) {
    if (depth >= max_recursive_depth) {
        *c = Color(0.0,0.0,0.0);
    } else {
        valarray<float> view = -1 * ray.direction;
        Color next_color = Color(0.0,0.0,0.0);
        Color cur_color = Color(0.0,0.0,0.0);
        Object* cur_object = objects[0];
        float epsilon = 0.01;
        float max_val = 1000;
        valarray<float> cur_pt = {0.0,0.0,0.0};
        float step_size = 0.1;
        float cur_t = 0;
        float d = 0;
        float prev_d = 0;
        bool decreasing = true;
        while(cur_t < max_val && decreasing) {
            ray.eval(cur_t, &cur_pt);
            cur_object->dist(cur_pt, &d);
            if (d < prev_d) {
                decreasing = false;          
            }
            prev_d = d;
            cur_t = cur_t + step_size;
        }
        Ray r = Ray(cur_pt, ray.direction);
        if (cur_object->t_hit(r, &cur_t)) {
            valarray<float> cord = {0,0,0};
            r.eval(cur_t, &cord);
            valarray<float> cur_norm = {0,0,0};
            cur_object->get_normal(cord, &cur_norm);
            shader.phong(cord, cur_norm, view, &cur_color, &cur_object->material);
            valarray<float> reflect = {0.0,0.0,0.0};
            reflectance(-1 * ray.direction, cur_norm, &reflect);
            Ray new_ray = Ray(cord, reflect);
            reflectance_harshil(new_ray, &next_color, (depth + 1.0));
            cur_color.add_color(next_color);
        }
        c->r = cur_color.r * reflectivity;
        c->g = cur_color.g * reflectivity;
        c->b = cur_color.b * reflectivity;
    }
    
}

/*
 *                  RAY CLASS
 * A parametrized ray defined by a point and a direction. 
 * Both are represented by valarrays for element-wise 
 * calculations. 
 *
 */

//Givees the xyz coordinate of the ray path at time t
void Ray::eval(float t, valarray<float>* cord) {
    valarray<float> new_cord = point + t * direction;
    cord->resize(3);
    cord->swap(new_cord);
}

/*
 *                      CAMERA CLASS
 * Camera class, which can take a sample's coordinates and create a ray from the eye 
 * location through this point in the image. It holds the eye position of the 
 * scene and given any sample it generates a ray from the eye postion through the 
 * xyz coordinate of the sample.
 *
 */

void Camera::generate_ray(valarray<float> world, Ray* r) {
    //cout << "CAMERA GENERATING RAY" << endl;
    r->point = world;
    valarray<float> d = world - eye_pos;
    normalize(&d);
    r->direction = d;
}

/*
 *                      SCENE CLASS
 * Given an eye position and image plane coordinates holds corresponding sampler, 
 * camera, raytracer and film. It's most important function is render, which uses
 * the sampler iterator to generate the color values for every pixel in our view 
 * window and then uses the film to write the image to a png file.
 *
 */

Scene::Scene(void) {
    UL.resize(3);
    UR.resize(3);
    LL.resize(3);
    LR.resize(3);
    UL = {1,1, 0};  
    UR = {-1,1,0};
    LL = {1,-1,0};
    LR = {-1,-1,0};
    resolution_x = x_resolution;
    resolution_y = y_resolution;
    sampler = Sampler();
    camera = Camera();
    raytracer = Raytracer();
    film = Film(x_resolution, y_resolution, 1);
}




// Clear all pixel buckets
void Scene::initialize(void) {
    cout << "Initializing Scene" << endl;
}

void Scene::render(void) {
    // This is the main rendering loop
    Sample sample = Sample();
    while(sampler.get_sample(&sample)) {
        Ray ray = Ray();
        valarray<float> world_cord;
        valarray<float> sample_cord = {sample.x, sample.y};
        screen_to_world(sample_cord, &world_cord);
        camera.generate_ray(world_cord, &ray);
        Color color = Color();
        raytracer.trace(ray, &color);
        film.commit(sample, color);
        //cout << "Completed SAMPLE (u,v): " << sample.x << ", " << sample.y << endl;
        //cout << "WORLD COORD (x,y,z): " << world_cord[0] << "," << world_cord[1] << "," << world_cord[2] << endl;
        //cout << "Ray Direction: "<< ray.direction[0] << "," << ray.direction[1] << "," << ray.direction[2] << endl;
    }
    film.write_image();
}

void Scene::screen_to_world(valarray<float> screen, valarray<float>* world) {
    float u = (screen[0]+0.5)/resolution_x;
    float v = (screen[1]+0.5)/resolution_y;
    //cout << "UV: " << u << "," << v << endl;
    valarray<float> final_point = (1.0 - u) * (v * LL + (1.0-v) * UL ) + (u) * (v * LR + (1.0 - v) * UR);
    //cout << "Final Point: " << final_point[0] << "," << final_point[1] << "," << final_point[2] << endl;
    world->resize(3);
    world->swap(final_point); 
    valarray<float> w = *world;
    //cout << "World Point: " << w[0] << "," << w[1] << "," << w[2] << endl;
}

bool loadOBJ(char *path, Material input_material) { 
    std::vector< valarray<float>> temp_vertices;
    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Can't read the OBJ file !\n");
        return false;
    } 
    while( 1 ) {
        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.
        //else : parse lineHeader
        if ( strcmp( lineHeader, "v" ) == 0 ) {
            float x,y,z; 
            fscanf(file, "%f %f %f\n", &x, &y, &z );
            valarray<float> vertex = {x,y,z};
            temp_vertices.push_back(vertex);
        } 
        else if ( strcmp( lineHeader, "f" ) == 0 ) {
            int v1,v2,v3;
            int matches = fscanf(file, "%d %d %d\n", &v1, &v2, &v3);
            if (matches != 3){
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }
            v1 = v1 -1;
            v2 = v2-1;
            v3 = v3 -1;
            Triangle* newtri = new Triangle(temp_vertices[v1], temp_vertices[v2], temp_vertices[v3], input_material);
            cout << "Triangle complete" << endl;
            objects.push_back(newtri);
            return true;
        }
    }
    cout << "OBJ DONE" << endl;
    return true;
} 

int main(int argc, char *argv[]) {
    cout << "Hello World." << endl;
    float r = 2;
    valarray<float> center;
    /*
    Material m = Material();
    for (int i = -10; i < 10; i+=10) {
        for(int j = -10; j < 10; j+=10) {
            for(int k = 10; k < 30; k+=10) {
                center = {i, j, k};
                objects.push_back(new Sphere(center, r, m));
            }
        }
    }
    */
    int i = 0;
    bool input_material = false; 
    Material last_material = default_material; 
    Scene* scn = new Scene();
    vector<valarray<float>> cur_LT = {};
    while( i + 1 <= argc ) {
        /*
         * The camera object is specified by a line of the following form:
         * cam ex ey ez llx lly llz lrx lry lrz ulx uly ulz urx ury urz
         * There will only be one camera line in a file.
         */
        if (strcmp(argv[i], "cam") == 0) { 
            //cout << "parsing camera" << endl;
            if (camera_set) { 
                i = i + 15;
            } else { 
                valarray<float> eye = {(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])};
                valarray<float> ll = {(float) atof(argv[i+4]), (float) atof(argv[i+5]), (float) atof(argv[i+6])};
                valarray<float> lr = {(float) atof(argv[i+7]), (float) atof(argv[i+8]), (float) atof(argv[i+9])};
                valarray<float> ul = {(float) atof(argv[i+10]), (float) atof(argv[i+11]), (float) atof(argv[i+12])};
                valarray<float> ur = {(float) atof(argv[i+13]), (float) atof(argv[i+14]), (float) atof(argv[i+15])};
                scn = new Scene(eye, ll, lr, ul, ur, x_resolution, y_resolution);
                i = i + 15;
            } 
        }
        /*
         * A material is specified by:
         * mat kar kag kab kdr kdg kdb ksr ksg ksb ksp krr krg krb
         * Where ka, kd, ks, and kr are the coefficients for ambient, diffuse, specular, and reflective.
         */
        else if (strcmp(argv[i], "mat") == 0) { 
            //cout << "parsing material" << endl;
            Color ka = Color((float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3]));
            Color kd = Color((float) atof(argv[i+4]), (float) atof(argv[i+5]), (float) atof(argv[i+6]));
            Color ks = Color((float) atof(argv[i+7]), (float) atof(argv[i+8]), (float) atof(argv[i+9]));
            float spu = (float) atof(argv[i+10]);
            float spv = (float) atof(argv[i+10]);
            Color kr = Color((float) atof(argv[i+11]), (float) atof(argv[i+12]), (float) atof(argv[i+13]));
            Material m1 = Material(ka, kd, ks, kr, spu, spv);
            input_material = true; 
            last_material = Material(ka, kd, ks, kr, spu, spv);
            i = i + 13;
        } 
        /*
         * A .obj file is specified by:
         * obj "file name”
         */
        else if (strcmp(argv[i], "obj") == 0) { 
            Material m; 
            if (input_material) {  
                m = last_material;   
            } else { 
                m = default_material;
            }
            if (!obj_parsed) { 
                obj_parsed = loadOBJ(argv[i+1], m); 
                if (!obj_parsed) { 
                    printf("Please Load a Correct OBJ File Format \n"); 
                    exit(1);
                } 
            } else { 
                printf("You have already parsed an obj \n"); 
            } 
            i = i + 1;
        } 
        /*
         * A point light source is specified by:
         * ltp px py pz r g b [falloff]
         */
        else if (strcmp(argv[i], "ltp") == 0) { 
            //cout << "parsing point light" << endl;
            valarray<float> point = {(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])};
            Color c = Color((float) atof(argv[i+4]), (float) atof(argv[i+5]), (float) atof(argv[i+6])); 
            Light *p_light = new Light(point, c, false, false);
            lights.push_back(p_light);
        } 
        /*
         * A directional light source is specified by:
         * ltd dx dy dz r g b
         */
        else if (strcmp(argv[i], "ltd") == 0) { 
            //cout << "parsing directional light" << endl;
            valarray<float> point = {(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])};
            Color c = Color((float) atof(argv[i+4]), (float) atof(argv[i+5]), (float) atof(argv[i+6])); 
            Light *d_light = new Light(point, c, true, false);
            lights.push_back(d_light);
        } 
        /*
         * An ambient light source is specified by:
         * lta r g b
         */
        else if (strcmp(argv[i], "lta") == 0) { 
            //cout << "parsing ambient light" << endl;
            valarray<float> point = {0.0,0.0,0.0};
            Color c = Color((float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])); 
            Light *a_light = new Light(point, c, false, true);
            lights.push_back(a_light);
        } 
        /*
         * A transformation is specified by any of the following:
         * xft tx ty tz
         * xfr rx ry rz
         * xfs sx sy sz
         * These refer in order to translation, rotation, and scaling. Rotations are exponential maps in degrees
         * Transformations apply cumulatively in the reverse order they appear in the file.
         *  Thus the following lines:
         *  xfr 45 0 0
         *  xfs 1 1 2
         *  xfr -45 0 0
         *  xft 1 2 3
         *  Would result in the following transformation matrix:
         *      [r 45 0 0]*[s 1 1 2]*[r -45 0 0]*[t 1 2 3]
         * When an object line is read, the current transformation will be applied to that object.
         * Transformations also apply to lights or the camera.
         * The current transformation may be reset to the identity with the following line:
         *      xfz
         */
        else if (strcmp(argv[i], "xfr") == 0) { 
            valarray<float> inst = {0,(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])}; 
            cur_LT.push_back(inst);
             
        } 
        else if (strcmp(argv[i], "xfs") == 0) { 
            valarray<float> inst = {1,(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])}; 
            cur_LT.push_back(inst);
             
             
        } 
        else if (strcmp(argv[i], "xft") == 0) { 
            valarray<float> inst = {2,(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])}; 
            cur_LT.push_back(inst);
             
             
        } 
        else if (strcmp(argv[i], "xfz") == 0) { 
            cur_LT = {}; 
             
        } 
        /*
         * A sphere is specified by:
         *   sph cx cy cz r
        
        For example:
        vector<enemyClass*> vectorOfEnemies;
        for (int i=0; i<numberOfEnemies; ++i)
        {
            enemyClass* someNewEnemyObjectPointer = new someNewEnemyObject();
            vectorOfEnemies.push_back(someNewEnemyObjectPointer);
            a whole bunch of other stuff I expect
        }
        make sure you deallocate the memory when you are done with your vector
        for (int i=0; i<vectorOfEnemies.size(); i++)
        {
        delete vectorOfEnemies[i];
        }
        */
        else if (strcmp(argv[i], "sph") == 0) { 
           //cout << "parsing sphere" << endl;
            Material m; 
            if (input_material) {  
                m = last_material;   
            } 
            valarray<float> c = {(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])};
            float radius = (float) atof(argv[i+4]);
            cout << "sphere KA r g b" << m.KA.r << m.KA.g << m.KA.b << endl;
            cout << "sphere KD r g b" << m.KD.r << m.KD.g << m.KD.b << endl;
            cout << "sphere KS r g b" << m.KS.r << m.KS.g << m.KS.b << endl;
            cout << "sphere KR r g b" << m.KR.r << m.KR.g << m.KR.b << endl;
            cout << "sphere SPU" << m.SPU << endl;
            Sphere* newsphere = new Sphere(c, radius, m);
            newsphere->lin_transform = cur_LT;
            objects.push_back(newsphere);
            i = i+4;
        } 
        /*
         * A triangle is specified by:
         *   tri ax ay az bx by bz cx cy cz
         */
        else if (strcmp(argv[i], "tri") == 0) { 
            Material m; 
            if (input_material) {  
                m = last_material;   
            } else { 
                m = default_material;
            } 
            valarray<float> a = {(float) atof(argv[i+1]), (float) atof(argv[i+2]), (float) atof(argv[i+3])};
            valarray<float> b = {(float) atof(argv[i+4]), (float) atof(argv[i+5]), (float) atof(argv[i+6])};
            valarray<float> c = {(float) atof(argv[i+7]), (float) atof(argv[i+8]), (float) atof(argv[i+9])};
            Triangle* newtri = new Triangle(a, b, c, m);
            newtri->lin_transform = cur_LT;
            objects.push_back(newtri);
            i = i+4;
        } 
        i = i + 1;
    }
    scn->initialize();
    scn->render();
    cout << "All done!" << endl;
}


