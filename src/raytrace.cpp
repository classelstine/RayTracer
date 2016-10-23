#include "raytrace.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
Scene *scn;
int x_resolution = 100;
int y_resolution = 100;
int samples_per_pix = 1;

// The Film class holds a bucket of Colors for each pixel value
Film::Film(int x_res, int y_res, int sample_rate) {
   pixel_buckets = vector<vector<vector<Color>>> (x_res, vector<vector<Color>>(y_res, vector<Color>(sample_rate, Color(0,0,0))));
}

Film::Film(void) {
    pixel_buckets = vector<vector<vector<Color>>> (100, vector<vector<Color>>(100, vector<Color>(1, Color(0,0,0))));
}

// Averages over each pixel bucks and writes the result to a screen
void Film::write_image(void) {
    cout << "ATTEMPTING TO WRITE IMAGE" << endl;
    char const* filename = "test_image";
    // DO NOT LEAVE THIS. THIS IS A TEST. 
    int w = 100;
    int h = 100;
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
            //    avg_pixels[cur_index] = 100;
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

void Camera::generate_ray(valarray<float> world, Ray* r) {
    //cout << "CAMERA GENERATING RAY" << endl;
    r->point = eye_pos;
    r->direction = world - eye_pos; 
}

// Currently is a dummy function which sets the color to 0.5
void Raytracer::trace(Ray r, Color *c) {
    //cout << "RAY TRACER TRACING" << endl;
    c->r = 0.5;
    c->g = 0;
    c->b = 0.5;
    // IF IS OBJ
    // for all obj polynomials, see if we hit, order them, figure out who is hit first
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
    max_x = 99;
    max_y = 99;
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
    resolution_x = 100;
    resolution_y = 100;
    sampler = Sampler();
    camera = Camera();
    raytracer = Raytracer();
    film = Film(100, 100, 1);
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

