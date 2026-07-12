#define RTVDB_IMPLEMENTATION
#include "rtvdb/rtvdb.h"

int main() {
    rtvdb::set_color(0.0f, 0.5f, 1.0f);
    rtvdb::triangle(
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f);

    rtvdb::set_color(1.0f, 0.5f, 0.0f, 0.5f);
    rtvdb::triangle(
        -1.5f, 0.0f, 0.5f,
        1.5f, -0.5f, 0.5f,
        1.0f, 1.0f, 0.5f);

    rtvdb::set_color(0.5f, 1.0f, 0.0f);
    rtvdb::set_line_radius(0.025f);
    rtvdb::line(-1.0f, -0.5f, 0.25f, 1.0f, 0.5f, 0.25f);

    rtvdb::set_color(1.0f, 0.0f, 0.0f);
    rtvdb::set_point_radius(0.05f);
    rtvdb::point(-0.5f, 0.5f, -0.5f);
    rtvdb::point(0.5f, -0.5f, 0.5f);

    return 0;
}
