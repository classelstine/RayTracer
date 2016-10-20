#include "raytrace.h"

using namespace std;
Scene *scn;
int x_resolution = 100;
int y_resolution = 100;
int samples_per_pix = 1;

// The Film class holds a bucket of Colors for each pixel value
Film::Film(int x_res, int y_res, int sample_rate) {
   pixel_buckets = vector<vector<vector<Color>>> (100, vector<vector<Color>>(100, vector<Color>(1, Color(0,0,0))));
}

// Averages over each pixel bucks and writes the result to a screen
void Film::write_image(void) {
    cout << "ATTEMPTING TO WRITE IMAGE" << endl;
}

// Add color c to the bucket of colors for sample s
void Film::commit(Sample s, Color c) {
    cout << "Committing color to film" << endl;
}

void Camera::generate_ray(Sample s, Ray* r) {
    cout << "CAMERA GENERATING RAY" << endl;
}

// Currently is a dummy function which sets the color to 0.5
void Raytracer::trace(Ray r, Color *c) {
    cout << "RAY TRACER TRACING" << endl;
    c->r = 0.5;
    c->g = 0.5;
    c->b = 0.5;
}

Sampler::Sampler(void) {
    current_px = 0;
    current_py = 0;
    max_x = 100;
    max_y = 100;
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
    return true
        

}

// Clear all pixel buckets
void Scene::initialize(void) {
    cout << "Initializing Scene" << endl;
}

void Scene::render(void) {
    // This is the main rendering loop
    Sample sample = Sample();
    while(!sampler.get_sample(&sample)) {
        Ray ray = Ray();
        camera.generate_ray(sample, &ray);
        Color color = Color();
        raytracer.trace(ray, &color);
        film.commit(sample, color);
        Sample sample = Sample();
    }
    film.write_image();
}

// Gets sample from screen coordinates. 
bool Sampler::get_sample(Sample *sample){   
    return true;
}


int main(int argc, char *argv[]) {
    cout << "Hello World." << endl;
    Scene* scn = new Scene();
    scn->initialize();
    scn->render();
    cout << "All done!" << endl;
}

