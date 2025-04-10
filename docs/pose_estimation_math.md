# 📐 Runway Pose Estimation - Mathematical Overview

## 1. Vanishing Point from RANSAC Lines

Given two lines in image space:
- \( y = m_1 x + b_1 \) (left)
- \( y = m_2 x + b_2 \) (right)

The vanishing point is their intersection:
\[
x_v = \frac{b_2 - b_1}{m_1 - m_2}, \quad y_v = m_1 x_v + b_1
\]

---

## 2. Estimating Heading Angle

Assuming known camera intrinsics \( K \), we compute:
\[
v = K^{-1} \cdot \begin{bmatrix} x_v \\ y_v \\ 1 \end{bmatrix}
\]

The horizontal angle (yaw) is:
\[
\theta = \arctan2(v_x, v_z)
\]

---

## 3. Estimating Lateral Offset (Optional)

If line spacing in 3D is known (e.g. 10 meters apart), we can back-project rays through both lines and intersect with ground plane to estimate lateral deviation.

---

## 4. Output

- Heading (yaw)
- Optional: lateral offset
- Confidence based on RANSAC line fit quality

---

## Note

This method assumes a flat ground plane and no roll/pitch. Accuracy depends on calibration and visibility of runway lanes.
