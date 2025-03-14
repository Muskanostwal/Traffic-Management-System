import cv2
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tracker import Tracker
import cv2.bgsegm

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Background Subtractor for density analysis
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Tracker for vehicle tracking
tracker = Tracker()

# Video capture
cap = cv2.VideoCapture('video_vehicles.mp4')

# Load class names for detection (ensure the file exists)
with open("cocoa.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Vehicle counters and constants
cy1, cy2 = 322, 368
offset = 6
counter_up, counter_down = 0, 0
vehicle_crossed = {}

# Speed analysis variables
speed_data = {}  # Dictionary to store each vehicle's speed {vehicle_id: [initial_time, initial_position, speed]}
real_world_distance = 20  # Distance between cy1 and cy2 in meters (real-world measurement)
car_speeds, truck_speeds = [], []

# Density analysis variables
vehicle_counts_per_minute = []
road_segment_length = 100  # Adjust this to your actual road segment length
start_time = time.time()
counter = 0

# Counters for pie chart
car_count, truck_count = 0, 0

# Pie chart setup
fig_pie, ax_pie = plt.subplots()
plt.ion()

# Graph setup for real-time density analysis
fig_density, ax_density = plt.subplots()
x_density_data, y_density_data = [], []
line_density, = ax_density.plot(x_density_data, y_density_data, label="Vehicles per minute")
ax_density.set_xlabel("Time (minutes)")
ax_density.set_ylabel("Vehicle Count")
ax_density.set_title("Real-Time Traffic Density Analysis")
ax_density.legend()

# Graph setup for speed analysis
fig_speed, ax_speed = plt.subplots()
x_speed_data, y_speed_data_car, y_speed_data_truck = [], [], []
line_car, = ax_speed.plot(x_speed_data, y_speed_data_car, label="Average Car Speed (m/s)")
line_truck, = ax_speed.plot(x_speed_data, y_speed_data_truck, label="Average Truck Speed (m/s)")
ax_speed.set_xlabel("Time (minutes)")
ax_speed.set_ylabel("Speed (m/s)")
ax_speed.set_title("Real-Time Speed Analysis")
ax_speed.legend()

# Function to get center of bounding box (used for vehicle counting)
def center_handle(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

# Function to calculate speed
def calculate_speed(vehicle_id, cx, cy, timestamp):
    if vehicle_id in speed_data:
        initial_time, initial_position = speed_data[vehicle_id][:2]
        time_diff = timestamp - initial_time
        distance = abs(cy - initial_position)  # Pixel distance

        # Convert pixel distance to real-world distance
        if cy1 < cy < cy2:
            speed_m_per_s = (real_world_distance / distance) / time_diff  # Speed in meters per second
            speed_data[vehicle_id].append(speed_m_per_s)
            return speed_m_per_s
    else:
        speed_data[vehicle_id] = [timestamp, cy]  # Store initial time and position
    return None

# Main processing loop
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 3 != 0:  # Process every third frame
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Vehicle detection using YOLOv8
    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    vehicle_list = []
    timestamp = time.time()  # Current time for speed calculation
    for _, row in px.iterrows():
        x1, y1, x2, y2, class_id = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        class_name = class_list[class_id]
        
        if 'car' in class_name or 'truck' in class_name:  # Track cars and trucks
            vehicle_list.append([x1, y1, x2, y2])
            speed = calculate_speed(_, (x1 + x2) // 2, (y1 + y2) // 2, timestamp)

            if speed:  # If speed is calculated
                if 'car' in class_name:
                    car_speeds.append(speed)
                elif 'truck' in class_name:
                    truck_speeds.append(speed)

        if 'car' in class_name:  # Count cars
            car_count += 1
        elif 'truck' in class_name:  # Count trucks
            truck_count += 1

    # Update tracker with detected vehicles
    bbox_id = tracker.update(vehicle_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Draw bounding box, ID, and center point
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0, (255, 255, 255), 2)

        # Draw counting lines
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 2)
        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 2)

        # Vehicle counting logic
        if cy1 - offset <= cy <= cy1 + offset:
            if vehicle_id not in vehicle_crossed:
                counter_down += 1
                vehicle_crossed[vehicle_id] = 'down'
        elif cy2 - offset <= cy <= cy2 + offset:
            if vehicle_id not in vehicle_crossed:
                counter_up += 1
                vehicle_crossed[vehicle_id] = 'up'

    # Vehicle counting display
    cv2.putText(frame, "Down Count: " + str(counter_down), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Up Count: " + str(counter_up), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    

    # Pie chart update every 1 minute
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        # Pie chart data
        labels = ['Cars', 'Trucks']
        sizes = [car_count, truck_count]
        colors = ['#ff9999', '#66b3ff']

        # Clear and update pie chart
        ax_pie.clear()
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_pie.set_title("Car and Truck Distribution in Last 1 Minute")

        plt.draw()  # Redraw the pie chart
        plt.pause(0.001)

        # Append data to line chart (time and vehicle count)
        x_density_data.append(len(x_density_data) + 1)  # Minutes passed
        y_density_data.append(car_count + truck_count)  # Total vehicles counted in the last minute

        # Redraw the density analysis line chart with updated data
        line_density.set_xdata(x_density_data)
        line_density.set_ydata(y_density_data)
        ax_density.relim()  # Adjust the axis limits based on the data
        ax_density.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)

        # Reset car and truck counts for the next minute
        car_count, truck_count = 0, 0
        start_time = time.time()

        # Speed analysis - calculate average speed for cars and trucks
        if len(car_speeds) > 0:
            avg_car_speed = np.mean(car_speeds)
        else:
            avg_car_speed = 0

        if len(truck_speeds) > 0:
            avg_truck_speed = np.mean(truck_speeds)
        else:
            avg_truck_speed = 0

        # Append speed data to line chart
        x_speed_data.append(len(x_speed_data) + 1)
        y_speed_data_car.append(avg_car_speed)
        y_speed_data_truck.append(avg_truck_speed)

        # Redraw the speed analysis line chart with updated data
        line_car.set_xdata(x_speed_data)
        line_car.set_ydata(y_speed_data_car)
        line_truck.set_xdata(x_speed_data)
        line_truck.set_ydata(y_speed_data_truck)
        ax_speed.relim()  # Adjust the axis limits based on the data
        ax_speed.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)

        # Clear speed lists for the next minute
        car_speeds.clear()
        truck_speeds.clear()

    # Display the result
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
