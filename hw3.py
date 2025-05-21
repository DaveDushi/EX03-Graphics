from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    epsilon = 1e-5

    def trace_ray(ray, depth):
        """Trace ``ray`` recursively and return the resulting color."""
        if depth > max_depth:
            return np.zeros(3)

        hit_obj, min_dist = ray.nearest_intersected_object(objects)
        if hit_obj is None or min_dist == np.inf:
            # Background color (black)
            return np.zeros(3)

        hit_point = ray.origin + ray.direction * min_dist

        # Determine normal based on object type
        if isinstance(hit_obj, Sphere):
            normal = normalize(hit_point - np.array(hit_obj.center))
        elif isinstance(hit_obj, Plane):
            normal = normalize(hit_obj.normal)
        elif isinstance(hit_obj, Triangle):
            normal = normalize(hit_obj.normal)
        else:
            # Generic object with 'normal' attribute
            normal = normalize(hit_obj.normal)

        # Start with ambient term
        color = hit_obj.ambient * ambient

        # Compute contribution of each light
        for light in lights:
            # Construct a light ray slightly above the surface to avoid self intersection
            light_ray = light.get_light_ray(hit_point + normal * epsilon)
            light_dir = light_ray.direction
            light_distance = light.get_distance_from_light(hit_point)

            # Shadow check
            shadow_obj, shadow_dist = light_ray.nearest_intersected_object(objects)
            if shadow_obj is not None and shadow_dist < light_distance - epsilon:
                continue

            intensity = light.get_intensity(hit_point)

            # Diffuse term
            diff = max(np.dot(normal, light_dir), 0.0)
            color += hit_obj.diffuse * diff * intensity

            # Specular term
            view_dir = normalize(camera - hit_point)
            reflect_dir = reflected(light_dir, normal)
            spec_angle = max(np.dot(view_dir, reflect_dir), 0.0)
            color += hit_obj.specular * (spec_angle ** hit_obj.shininess) * intensity

        # Reflection
        if hit_obj.reflection > 0 and depth < max_depth:
            reflect_dir = reflected(ray.direction, normal)
            reflect_origin = hit_point + normal * epsilon
            reflect_color = trace_ray(Ray(reflect_origin, reflect_dir), depth + 1)
            color = (1 - hit_obj.reflection) * color + hit_obj.reflection * reflect_color

        return color

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = trace_ray(ray, 0)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
