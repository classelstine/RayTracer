#include "raytrace.h"

using namespace std;

int x_resolution = 100;
int y_resolution = 100;
int samples_per_pix = 1;

// The Film class holds a bucket of Colors for each pixel value
Film::Film(int x_res, int y_res, int sample_rate) {
   pixel_buckets = vector<vector<vector<Color>>> (, vector<vector<Color>>(, vector<Color>(0,0,0))); 
}

// Averages over each pixel bucks and writes the result to a screen
void Film::write_image(void) {
    cout << "ATTEMPTING TO WRITE IMAGE" << endl;
}

void Scene::render(void) {
    // This is the main rendering loop
}

// Gets sample from screen coordinates. 
bool Sampler::get_sample(Sample *sample){   
    return true;
}


int main(int argc, char *argv[]) {
    cout << "Hello World." << endl;
    Scene scn = new Scene();
    scn->initialize();
    scn->render();
    cout << "All done!" << endl;
}

