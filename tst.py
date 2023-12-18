import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def update_max_object_info(ellipse_center_x, ellipse_center_y, ellipse_major_axis, ellipse_minor_axis, max_x_object_info, max_y_object_info):
    # التحقق من الكائن ذو أكبر قيمة X
    if max_x_object_info is None or ellipse_center_x > max_x_object_info[0][0]:
        max_x_object_info = ((ellipse_center_x, ellipse_center_y), (ellipse_major_axis // 2, ellipse_minor_axis // 2))
    # التحقق من الكائن ذو أكبر قيمة Y
    if max_y_object_info is None or ellipse_center_y > max_y_object_info[0][1]:
        max_y_object_info = ((ellipse_center_x, ellipse_center_y), (ellipse_major_axis // 2, ellipse_minor_axis // 2))
    return max_x_object_info, max_y_object_info


def calculate_and_print_distances(max_x_object_info, max_y_object_info,sorted_stats,fenster_groesse, ellipses, labels, image):     
    for i, stats_entry in enumerate(sorted_stats, start=1):
        x, y, breite, hoehe, area = stats_entry
        if labels[y, x] != 0:
            fenster = image[y - fenster_groesse:y + fenster_groesse, x - fenster_groesse:x + fenster_groesse]
            hellster_pixel = np.max(fenster)
            maske = fenster >= hellster_pixel / 2
            indizes = np.argwhere(maske)
            if np.any(maske):
                breite = np.max(indizes[:, 1]) - np.min(indizes[:, 1])
                hoehe = np.max(indizes[:, 0]) - np.min(indizes[:, 0])
        center_x = x + breite // 2
        center_y = y + hoehe // 2
        radiating_pixel_value = np.max(image[center_y - fenster_groesse:center_y + fenster_groesse,
                                       center_x - fenster_groesse:center_x + fenster_groesse])
        mask = image[center_y - hoehe // 2:center_y + hoehe // 2,
                 center_x - breite // 2:center_x + breite // 2] >= radiating_pixel_value / 2
        if np.any(mask):
            indices = np.argwhere(mask)
            ellipse_center_x = center_x + (np.max(indices[:, 1]) + np.min(indices[:, 1])) // 2 - breite // 2
            ellipse_center_y = center_y + (np.max(indices[:, 0]) + np.min(indices[:, 0])) // 2 - hoehe // 2
            ellipse_major_axis = np.max(indices[:, 1]) - np.min(indices[:, 1])
            ellipse_minor_axis = np.max(indices[:, 0]) - np.min(indices[:, 0])
            max_x_object_info, max_y_object_info = update_max_object_info(
                ellipse_center_x, ellipse_center_y, ellipse_major_axis, ellipse_minor_axis, max_x_object_info, max_y_object_info)                
    # حساب وطباعة البعد بين أول كائن وآخر كائن على المحور الأفقي والعمودي
    if max_x_object_info is not None and max_y_object_info is not None:
        first_object_coordinates = max_x_object_info[0]
        last_object_coordinates = max_y_object_info[0]
        vertical_distance = abs(last_object_coordinates[1] - first_object_coordinates[1]) 
        horizontal_distance = abs(last_object_coordinates[0] - first_object_coordinates[0]) 
        print(f"Vertical Distance between the first and last object: {vertical_distance:.2f} pixel")
        print(f"Horizontal Distance between the first and last object: {horizontal_distance:.2f} pixel")
        # حساب وطباعة حجم البكسل بناءً على البعد الأفقي والعمودي
        pixel_size_x_mm = 27.5 / horizontal_distance
        pixel_size_y_mm = 13.5 / vertical_distance
        print(f"Pixel Size (mm) - Width: {pixel_size_x_mm:.4f} mm, High: {pixel_size_y_mm:.4f} mm")
        return pixel_size_x_mm, pixel_size_y_mm
    
        
def select_image_and_process():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.jfif")])
    if file_path:
        image = cv2.imread(file_path)
        process_image(image)


def process_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    kernel = np.ones((15, 25), np.uint8)
    dilated_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
    image[dilated_mask > 0] = [0, 0, 0]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    # ترتيب الكائنات بناءً على القيمة الأفقية (x)
    sorted_stats_x = sorted(stats[1:], key=lambda x: x[0], reverse=True)
    # ترتيب الكائنات بناءً على القيمة العمودية (y)
    sorted_stats_y = sorted(stats[1:], key=lambda x: x[1], reverse=True)
    
    
    # تحديد الكائن ذو أكبر قيمة x ككائن رقم 1
    max_x_object_info = ((sorted_stats_x[0][0], sorted_stats_x[0][1]), (sorted_stats_x[0][2] // 2, sorted_stats_x[0][3] // 2))
    # تحديد الكائن ذو أكبر قيمة y ككائن صاحب الرقم الأخير
    max_y_object_info = ((sorted_stats_y[-1][0], sorted_stats_y[-1][1]), (sorted_stats_y[-1][2] // 2, sorted_stats_y[-1][3] // 2))
    sorted_stats = sorted(stats[1:], key=lambda x: (x[0], x[1]), reverse=True)
    fenster_groesse = 8
    ellipses = []
    max_x_object_info = None
    max_y_object_info = None    
    pixel_size_x_mm, pixel_size_y_mm = calculate_and_print_distances(max_x_object_info, max_y_object_info, sorted_stats, fenster_groesse, ellipses, labels, image)
    for i, stats_entry in enumerate(sorted_stats, start=1):
        x, y, breite, hoehe, area = stats_entry
        if labels[y, x] != 0:
            fenster = image[y - fenster_groesse:y + fenster_groesse, x - fenster_groesse:x + fenster_groesse]
            hellster_pixel = np.max(fenster)
            maske = fenster >= hellster_pixel / 2
            indizes = np.argwhere(maske)
            if np.any(maske):
                breite = np.max(indizes[:, 1]) - np.min(indizes[:, 1])
                hoehe = np.max(indizes[:, 0]) - np.min(indizes[:, 0])
        center_x = x + breite // 2
        center_y = y + hoehe // 2
        radiating_pixel_value = np.max(image[center_y - fenster_groesse:center_y + fenster_groesse,
                                       center_x - fenster_groesse:center_x + fenster_groesse])
        mask = image[center_y - hoehe // 2:center_y + hoehe // 2,
                 center_x - breite // 2:center_x + breite // 2] >= radiating_pixel_value / 2
        if np.any(mask):
            indices = np.argwhere(mask)
            ellipse_center_x = center_x + (np.max(indices[:, 1]) + np.min(indices[:, 1])) // 2 - breite // 2
            ellipse_center_y = center_y + (np.max(indices[:, 0]) + np.min(indices[:, 0])) // 2 - hoehe // 2
            ellipse_major_axis = np.max(indices[:, 1]) - np.min(indices[:, 1])
            ellipse_minor_axis = np.max(indices[:, 0]) - np.min(indices[:, 0])
            lateral_resolution_mm = ellipse_major_axis * pixel_size_x_mm
            axial_resolution_mm = ellipse_minor_axis * pixel_size_y_mm    
            if lateral_resolution_mm <= 3 and lateral_resolution_mm >= 0.0:
                ellipses.append(((ellipse_center_x, ellipse_center_y), (ellipse_major_axis // 2, ellipse_minor_axis // 2)))
                print(f"Object {i} Object at Point ({x}, {y}) with Width {ellipse_major_axis} and Height {ellipse_minor_axis}")
                print(f"Lateral Resolution: {lateral_resolution_mm:.2f} mm")
                print(f"Axial Resolution: {axial_resolution_mm:.2f} mm")   
                cv2.putText(image, str(i), (ellipse_center_x - 5, ellipse_center_y + ellipse_minor_axis // 2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    for ellipse_params in ellipses:
        cv2.ellipse(image, ellipse_params[0], ellipse_params[1], 0, 0, 360, (255, 255, 0), 2)

    cv2.imshow('Ultraschallbild', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def main():
    select_image_and_process()

if __name__ == "__main__":
    main()