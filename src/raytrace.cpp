#include "raytrace.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
Scene *scn;
int x_resolution = 1000;
int y_resolution = 1000;
int samples_per_pix = 1;
valarray<float> c1 = {0.0, 0.0, 10.0};
Color KA = Color(0.6, 0.2, 0.7);
Color KD = Color(0.1, 0.8, 0.4);
Color KS = Color(0.3, 0.7, 0.2);
float SPU = 2;
float SPV = 2;
Sphere* s1 = new Sphere(c1, 2.0, KA, KD, KS, SPU, SPV); 
Object* objects[] = {s1};


void Ray::eval(float t, valarray<float>* cord) {
    valarray<float> new_cord = point + t * direction;
    cord->resize(3);
    cord->swap(new_cord);
}

// The Film class holds a bucket of Colors for each pixel value
Film::Film(int x_res, int y_res, int sample_rate) {
   pixel_buckets = vector<vector<vector<Color>>> (x_res, vector<vector<Color>>(y_res, vector<Color>(sample_rate, Color(0,0,0))));
}

Film::Film(void) {
    pixel_buckets = vector<vector<vector<Color>>> (1000, vector<vector<Color>>(1000, vector<Color>(1, Color(0,0,0))));
}

// Averages over each pixel bucks and writes the result to a screen
void Film::write_image(void) {
    cout << "ATTEMPTING TO WRITE IMAGE" << endl;
    char const* filename = "test_image";
    // DO NOT LEAVE THIS. THIS IS A TEST. 
    int w = 1000;
    int h = 1000;
    int stride = 3*w;
    int comp = 3;
    int cur_index;
    uint8_t avg_pixels[w * h * 3];
    // NEEDS TO BE REPLACED WITH FINDING AVERGE FOR EACH PIXEL BUCKET
    for(int row = 0; row < h; row++) {
        for(int col = 0; col < w; col++) {
            int p_index = row*w + col;
            cur_index = 3*p_index;
            Color cur_color = pixel_buckets[row][col][0];
            avg_pixels[cur_index] = cur_color.r * 255;
            avg_pixels[cur_index+1] = cur_color.g * 255;
            avg_pixels[cur_index+2] = cur_color.b * 255;
            //avg_pixels[cur_index] = 0;
            //avg_pixels[cur_index+1] = 0;
            //avg_pixels[cur_index+2] = 0;
            //float d = sqrt(pow((h/2) -row, 2) + pow((w/2) -col, 2));
            //if (d < 50) {
            //    avg_pixels[cur_index] = 1000;
            //}
        }
    }

    stbi_write_png(filename, w, h, comp, &avg_pixels, stride);
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

bool Sphere::t_hit(Ray ray, float* t) {
    //cout<< "in sphere t_hit" << endl;
    valarray<float> d = ray.direction;
    valarray<float> e = ray.point;
    float discriminant = pow(dot(d,(e - center)), 2) - (dot(d, d)) * (dot(e - center, e - center) - pow(radius, 2));
    if(discriminant < 0) {
        *t = -1;
        return false;
    }
    float numerator = -1 * (dot(d, (e-center)));
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


void reflectance(valarray<float> light_source, valarray<float> normal, valarray<float> *reflectance) { 
    *reflectance = -1*light_source + 2*dot(light_source, normal)*normal;
    normalize(reflectance);
}

float find_specular_power(valarray<float> normal, valarray<float> view, valarray<float> light_vec, Object* obj) {
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

void Shader::phong(valarray<float> point, valarray<float> normal, valarray<float> view, Color *c, Object *obj) {
    Color tmp_pixel_color = Color(0.0, 0.0, 0.0);
    valarray<float> cur_point = point;
    Color ambient = Color(0.0, 0.0, 0.0);
    Color diffuse = Color(0.0, 0.0, 0.0);
    Color specular = Color(0.0, 0.0, 0.0);
    /*
    cout << "r: " << obj->KA.r << obj->KA.g <<endl;
    cout << obj->KA.g << endl;
    cout << obj->KA.b << endl;
    */

    for(int d = 0; d < lights.size(); d++) {
      Light cur_light = lights[d];
      valarray<float> light_vec = {0.0,0.0,0.0};
      cur_light.light_vector(point, &light_vec);
      Color light_col = cur_light.color;
      valarray<float> reflect = {0.0,0.0,0.0};
      reflectance(light_vec, normal, &reflect);
      Color new_ambient = Color();
      mult_color(obj->KA, light_col, &new_ambient);
      ambient.add_color(new_ambient);
      Color new_diffuse = Color();
      Color diff1 = Color();
      float l_n = dot(light_vec, normal);
      float positive_dot = max(l_n,(float)  0.0);
      mult_color(obj->KD, light_col, &diff1);
      scale_color(positive_dot, diff1, &new_diffuse);
      diffuse.add_color(new_diffuse);
      Color new_specular = Color();
      Color spec1 = Color();
      float ref_view = dot(reflect, view);
      float mx = max(ref_view, (float) 0.0);
      float power = find_specular_power(normal, view, light_vec, obj);
      float tmp = pow(mx, power);
      scale_color(tmp, obj->KS, &spec1);
      mult_color(spec1, light_col, &new_specular);
      specular.add_color(new_specular);
    }
  tmp_pixel_color.add_color(ambient); 
  tmp_pixel_color.add_color(diffuse); 
  cout << "diffuse r,g,b: " << diffuse.r << " " << diffuse.g << " " << diffuse.b << endl;
  tmp_pixel_color.add_color(specular); 
  cout << "specular r,g,b: " << specular.r << " " << specular.g << " " << specular.b << endl;
  c->r = tmp_pixel_color.r;
  c->g = tmp_pixel_color.g;
  c->b = tmp_pixel_color.b;
}

void Camera::generate_ray(valarray<float> world, Ray* r) {
    //cout << "CAMERA GENERATING RAY" << endl;
    r->point = eye_pos;
    r->direction = world - eye_pos; 
}

// Currently is a dummy function which sets the color to 0.5
void Raytracer::trace(Ray r, Color *c) {
    //cout << "RAY TRACER TRACING" << endl;
    valarray<float> view = -1 * r.direction;
    normalize(&view);
    // IF IS OBJ
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
            shader.phong(cord, cur_norm, view, c, cur_object);    
        }
    }
    // STEP 1: FIND HIT POINT
    //STEP 2: CALCULATE NORMAL
    // for each light source
    //  STEP 3: SEND SHADOW RAY
    //  STEP 4: SHADE
    //  STEP 5: add recursive step
    
}



Sampler::Sampler(void) {
    current_px = 0;
    current_py = 0;
    max_x = 999;
    max_y = 999;
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

Scene::Scene(void) {
    UL.resize(3);
    UR.resize(3);
    LL.resize(3);
    LR.resize(3);
    UL = {1,1, 0};  
    UR = {-1,1,0};
    LL = {1,-1,0};
    LR = {-1,-1,0};
    resolution_x = 1000;
    resolution_y = 1000;
    sampler = Sampler();
    camera = Camera();
    raytracer = Raytracer();
    film = Film(1000, 1000, 1);
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
        //cout << "SAMPLE (u,v): " << sample.x << ", " << sample.y << endl;
        //cout << "WORLD COORD (x,y,z): " << world_cord[0] << "," << world_cord[1] << "," << world_cord[2] << endl;
        //cout << "Ray Direction: "<< ray.direction[0] << "," << ray.direction[1] << "," << ray.direction[2] << endl;
    }
    film.write_image();
}

void Scene::screen_to_world(valarray<float> screen, valarray<float>* world) {
    float u = (screen[0]+0.5)/resolution_x;
    float v = (screen[1]+0.5)/resolution_y;
    //cout << "UV: " << u << "," << v << endl;
    valarray<float> final_point = u * (v * LL + (1.0-v) * UL ) + (1.0 - u) * (v * LR + (1.0 - v) * UR);
    //cout << "Final Point: " << final_point[0] << "," << final_point[1] << "," << final_point[2] << endl;
    world->resize(3);
    world->swap(final_point); 
    valarray<float> w = *world;
    //cout << "World Point: " << w[0] << "," << w[1] << "," << w[2] << endl;

}

int main(int argc, char *argv[]) {
    cout << "Hello World." << endl;
    Scene* scn = new Scene();
    scn->initialize();
    scn->render();
    cout << "All done!" << endl;
}

