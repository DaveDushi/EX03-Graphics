import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    """Return the reflection of ``vector`` around ``axis``.

    Parameters
    ----------
    vector : array_like
        Incoming vector that should be reflected.
    axis : array_like
        Surface normal (the axis we reflect about).

    Returns
    -------
    numpy.ndarray
        The reflected vector.
    """

    # Ensure the axis is normalized to avoid scaling the result
    n = normalize(axis)
    # Compute the reflection using the formula r = v - 2*(v·n)*n
    reflected_vec = vector - 2 * np.dot(vector, n) * n
    return reflected_vec

## Lights


class LightSource:
    def __init__(self, intensity):
        # store intensity as numpy array to allow element-wise operations
        self.intensity = np.array(intensity, dtype=float)


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        # a directional light is defined only by its incoming direction
        # we normalize to keep subsequent computations stable
        self.direction = normalize(np.array(direction))

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        """Return a ray from ``intersection_point`` towards the light."""
        # For directional light, all rays travel in the same direction
        # (opposite to the light's direction vector).
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):

        # Directional light has no specific origin, hence the distance is
        # conceptually infinite. This can be used to disable attenuation.
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # Intensity is constant everywhere for directional lights
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        # Euclidean distance from the intersection point to the light position
        return np.linalg.norm(self.position - intersection)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # calculate distance between light source and intersection
        d = self.get_distance_from_light(intersection)
        attenuation = 1.0 / (self.kc + self.kl * d + self.kq * d * d)
        return self.intensity * attenuation


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        # store spotlight position and direction
        self.position = np.array(position, dtype=float)
        # normalize the incoming direction of the spot light
        self.direction = normalize(np.array(direction, dtype=float))
        # attenuation coefficients
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        # ray from the intersection point towards the light position
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        # distance attenuation similar to a point light
        d = self.get_distance_from_light(intersection)
        attenuation = 1.0 / (self.kc + self.kl * d + self.kq * d * d)

        # spotlight factor based on the angle between light direction and the
        # vector from the light to the intersection point
        to_point = normalize(intersection - self.position)
        spot_factor = max(np.dot(self.direction, to_point), 0.0)

        return self.intensity * attenuation * spot_factor


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf

        # Iterate over all objects and keep track of the closest intersection
        for obj in objects:
            hit = obj.intersect(self)
            if hit is None:
                continue
            # `intersect` is expected to return a tuple (distance, obj)
            distance, hit_obj = hit
            if distance < min_distance:
                min_distance = distance
                nearest_object = hit_obj

        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        # convert lists to numpy arrays for vector math
        self.ambient = np.array(ambient, dtype=float)
        self.diffuse = np.array(diffuse, dtype=float)
        self.specular = np.array(specular, dtype=float)
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        """Compute and return the unit normal of the triangle."""
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        normal = np.cross(edge1, edge2)
        return normalize(normal)

    def intersect(self, ray: Ray):
        """Return the intersection of ``ray`` with the triangle."""
        epsilon = 1e-6

        edge1 = self.b - self.a
        edge2 = self.c - self.a

        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < epsilon:
            return None

        f = 1.0 / a
        s = ray.origin - self.a
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)
        if t > epsilon:
            return t, self
        return None

class Diamond(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        # TODO
        return l

    def apply_materials_to_triangles(self):
        # TODO
        pass

    def intersect(self, ray: Ray):
        # TODO
        pass

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        """Return the intersection distance ``t`` and the object if the ray
        intersects the sphere.

        Parameters
        ----------
        ray : Ray
            Ray for which we want to find the intersection with the sphere.

        Returns
        -------
        tuple or None
            ``(t, self)`` where ``t`` is the distance from the ray origin to
            the intersection point. ``None`` is returned when there is no
            intersection in front of the ray origin.
        """

        # Vector from ray origin to the sphere center
        oc = ray.origin - np.array(self.center)

        # Coefficients of the quadratic equation a*t^2 + b*t + c = 0
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, oc)
        c = np.dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)

        # Find the nearest positive intersection
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t = None
        if t1 > 1e-6 and t2 > 1e-6:
            t = min(t1, t2)
        elif t1 > 1e-6:
            t = t1
        elif t2 > 1e-6:
            t = t2

        if t is not None:
            return t, self
        return None

