from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import least_squares

app = Flask(__name__)
C = 299792458.0  # Speed of light in m/s


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
                result_data = {
                    "x": f"{X_rec:.3f}",
                    "y": f"{Y_rec:.3f}",
                    "z": f"{Z_rec:.3f}",
                    "bias_meters": f"{clock_bias:.3f}",
                    "bias_seconds": f"{clock_bias / C:.9f}"
                }
            else:
                error = f"Optimization failed to converge: {result.message}"

        except Exception as e:
            error = f"Invalid input or calculation error: {str(e)}"

    return render_template("index.html", result=result_data, error=error)


if __name__ == "__main__":
    app.run(debug=True)