from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import least_squares

app = Flask(__name__)
C = 299792458.0  # Speed of light in m/s


def ecef_to_geodetic(x, y, z):
    """
    Converts ECEF Cartesian coordinates (x, y, z) in meters
    to WGS-84 Latitude (deg), Longitude (deg), and Altitude (meters).
    """
    a = 6378137.0          # Semi-major axis in meters
    f = 1.0 / 298.257223563 # Flattening
    b = a * (1.0 - f)      # Semi-minor axis
    e2 = 2.0 * f - f**2    # First eccentricity squared
    ep2 = (a**2 - b**2) / b**2 # Second eccentricity squared

    p = np.sqrt(x**2 + y**2)
    if p == 0:
        lat = np.pi / 2.0 if z > 0 else -np.pi / 2.0
        lon = 0.0
        alt = abs(z) - b
        return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)

    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    lon = np.arctan2(y, x)

    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


@app.route("/", methods=["GET", "POST"])
def index():
    result_data = None
    error = None

    if request.method == "POST":
        try:
            satellites = []
            pseudoranges = []

            # Collect data for 4 satellites from form inputs
            for i in range(1, 5):
                x = float(request.form[f"sat{i}_x"])
                y = float(request.form[f"sat{i}_y"])
                z = float(request.form[f"sat{i}_z"])
                travel_time = float(request.form[f"sat{i}_time"])

                satellites.append([x, y, z])
                pseudoranges.append(travel_time * C)

            satellites = np.array(satellites)
            pseudoranges = np.array(pseudoranges)

            # Residual function
            def residuals(state):
                X, Y, Z, b = state
                res = []
                for i in range(4):
                    geometric_dist = np.sqrt(
                        (X - satellites[i, 0]) ** 2 +
                        (Y - satellites[i, 1]) ** 2 +
                        (Z - satellites[i, 2]) ** 2
                    )
                    res.append(geometric_dist + b - pseudoranges[i])
                return res

            # Initial guess
            initial_guess = [
                np.mean(satellites[:, 0]),
                np.mean(satellites[:, 1]),
                np.mean(satellites[:, 2]),
                0.0
            ]

            # Solve
            result = least_squares(residuals, initial_guess)

            if result.success:
                X_rec, Y_rec, Z_rec, clock_bias = result.x

                # Compute WGS-84 coordinates from your ECEF solution
                lat, lon, alt = ecef_to_geodetic(X_rec, Y_rec, Z_rec)

                result_data = {
                    "x": f"{X_rec:.3f}",
                    "y": f"{Y_rec:.3f}",
                    "z": f"{Z_rec:.3f}",
                    "bias_meters": f"{clock_bias:.3f}",
                    "bias_seconds": f"{clock_bias / C:.9f}",
                    "lat": f"{lat:.6f}",
                    "lon": f"{lon:.6f}",
                    "alt": f"{alt:.2f}"
                }
            else:
                error = f"Optimization failed to converge: {result.message}"

        except Exception as e:
            error = f"Invalid input or calculation error: {str(e)}"

    return render_template("index.html", result=result_data, error=error)


if __name__ == "__main__":
    app.run(debug=True)