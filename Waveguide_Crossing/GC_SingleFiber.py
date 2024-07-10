import numpy as np
from phidl import quickplot as qp
from phidl import Device
import phidl.geometry as pg
import math



def make_grating_coupler(target_length, duty_cycle, pitch, radius, y_span, L_extra, waveguide_width,
                         waveguide_length):

  gc = Device('grating coupler')

  arc_extension = 1
  n_periods = math.ceil(target_length/pitch)
  fill_width = pitch*duty_cycle
  etch_width = pitch*(1-duty_cycle)
  total_length = n_periods*pitch + etch_width
  theta = math.asin(((0.5*y_span)/radius))*(180/math.pi)

  inner_rad_ia = radius*math.cos((theta*math.pi)/180)
  outer_rad_ia = radius
  #input_arc_radius = (inner_rad_ia + outer_rad_ia)/2
  #input_arc_width = outer_rad_ia - inner_rad_ia

  #input_arc = pg.arc(radius = input_arc_radius, width = input_arc_width, theta = 2*theta,
  #                 start_angle = -theta, angle_resolution = 2.5, layer = 2)

  #gc.add_ref(input_arc)

  inner_rad_oa = radius + total_length
  outer_rad_oa = radius + total_length + L_extra
  #output_arc_radius = (inner_rad_oa + outer_rad_oa)/2
  #output_arc_width = outer_rad_oa - inner_rad_oa

  #output_arc = pg.arc(radius = output_arc_radius, width = output_arc_width, theta = 2*theta,
  #                    start_angle = -theta, angle_resolution = 2.5, layer = 2)

  #gc.add_ref(output_arc)
  arc_radius = (inner_rad_ia + outer_rad_oa)/2
  arc_width = outer_rad_oa - inner_rad_ia

  arc = pg.arc(radius = arc_radius, width = arc_width, theta = 2*theta,
                        start_angle = -theta, angle_resolution = 0.025, layer = 3)
  
  gc.add_ref(arc)


  for i in range(n_periods+1):
    inner_rad_ring = radius + pitch*(i)
    #print(inner_rad_ring)
    outer_rad_ring = radius + pitch*(i) +  etch_width
    #print(outer_rad_ring)
    ring_radius = (inner_rad_ring + outer_rad_ring)/2
    ring_width = outer_rad_ring - inner_rad_ring
    arc_length = (theta*(math.pi/180))*ring_radius
    #print(arc_length)
    new_arc_length = arc_length + arc_extension
    #print(new_arc_length)
    new_theta = (new_arc_length/ring_radius)*(180/math.pi)
    #print(new_theta)


    ring = pg.arc(radius = ring_radius, width = ring_width, theta = 2*new_theta,
                        start_angle = -new_theta, angle_resolution = 0.025, layer = 11)

    gc.add_ref(ring)

  taper = pg.taper(length = radius*math.cos((theta*math.pi)/180), width1 = waveguide_width, width2 = y_span,
                   port = None, layer = 3)
  gc.add_ref(taper)

  # waveguide = pg.rectangle(size = (waveguide_length, waveguide_width), layer = 3)
  # waveguide.movex(-waveguide_length)
  # waveguide.movey(-waveguide_width/2)

  # gc.add_ref(waveguide)

  #extended_triangle
  angle_a = (180 - 2*theta)/2
  alpha = 180 - angle_a
  beta = 12   #parameter for the extended triangle
  gamma = 180 - alpha - beta

  y1 = math.sin(math.radians(theta))*outer_rad_oa
  y2 = -y1
  x1 = math.cos(math.radians(theta))*outer_rad_oa
  x2 = x1

  x3, y3 = find_third_point(x1, y1, x2, y2, alpha, beta, gamma)
  y3 = -y3

  point1 = (x1, y1)
  point2 = (x2, y2)
  point3 = (x3, y3)

  # Draw the triangle using the defined points
  triangle_device = draw_triangle([point1, point2, point3])

  gc.add_ref(triangle_device)

  #Polygon device
  # poly_offset = 5
  # poly_length = 20
  # poly_width = 5

  # lower_poly_y1 = - poly_offset
  # lower_poly_x1 = 0

  # lower_poly_x2 = lower_poly_x1 + poly_length

  # dist = poly_length*math.tan((math.pi/180)*(18.92/2))
  # lower_poly_y2 = lower_poly_y1 - dist

  # lower_poly_x3 = lower_poly_x1
  # lower_poly_x4 = lower_poly_x2
  # lower_poly_y3 = lower_poly_y1 - poly_width
  # lower_poly_y4 = lower_poly_y3

  # upper_poly_y1 = poly_offset
  # upper_poly_x1 = 0

  # upper_poly_x2 = upper_poly_x1 + poly_length

  # dist = poly_length*math.tan((math.pi/180)*(18.92/2))
  # upper_poly_y2 = upper_poly_y1 + dist

  # upper_poly_x3 = upper_poly_x1
  # upper_poly_x4 = upper_poly_x2
  # upper_poly_y3 = upper_poly_y1 + poly_width
  # upper_poly_y4 = upper_poly_y3

  # upper_poly1 = (upper_poly_x1, upper_poly_y1)
  # upper_poly2 = (upper_poly_x2, upper_poly_y2)
  # upper_poly3 = (upper_poly_x3, upper_poly_y3)
  # upper_poly4 = (upper_poly_x4, upper_poly_y4)
   
  # lower_poly1 = (lower_poly_x1, lower_poly_y1)
  # lower_poly2 = (lower_poly_x2, lower_poly_y2)
  # lower_poly3 = (lower_poly_x3, lower_poly_y3)
  # lower_poly4 = (lower_poly_x4, lower_poly_y4)

  # upper_poly = draw_polygon([upper_poly1, upper_poly2, upper_poly4, upper_poly3])
  # lower_poly = draw_polygon([lower_poly1, lower_poly2, lower_poly4, lower_poly3])

  # upper_polygon_ref = gc.add_ref(upper_poly)
  # lower_polygon_ref = gc.add_ref(lower_poly)


  #Central_marker
  marker_length = 4
  marker_width = 1
  marker_center = (outer_rad_ia + inner_rad_oa)/2
  marker1 = pg.rectangle(size = (marker_length, marker_width), layer = 999)  
  marker2 = pg.rectangle(size = (marker_width, marker_length), layer = 999) 
  marker1.movex(-marker_length/2)
  marker1.movex(marker_center)
  marker1.movey(- marker_width/2)
  marker2.movex(marker_center)
  marker2.movex(- marker_width/2)
  marker2.movey(- (marker_length/2))
  marker1_ref = gc.add_ref(marker1)
  marker2_ref = gc.add_ref(marker2)


  return gc



def find_third_point(x1, y1, x2, y2, alpha, beta, gamma):
    # Convert angles to radians
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    # Calculate the length of side AB
    AB = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Use the Law of Sines to find the lengths of the other sides
    BC = (AB * math.sin(alpha)) / math.sin(gamma)
    AC = (AB * math.sin(beta)) / math.sin(gamma)

    # Find the angle between AB and the x-axis
    angle_AB_x_axis = math.atan2(y2 - y1, x2 - x1)

    # Find the angle for AC relative to AB
    angle_A = angle_AB_x_axis + alpha

    # Calculate the coordinates of point C
    x3 = x1 + AC * math.cos(angle_A)
    y3 = y1 + AC * math.sin(angle_A)

    return x3, y3



def draw_triangle(points):
    # Create a new device
    D = Device('triangle')

    # Add a polygon to the device with the provided points
    D.add_polygon(points, layer = 3)

    return D



def draw_polygon(points):
    # Create a new device
    D = Device('polygon')

    # Add a polygon to the device with the provided points
    D.add_polygon(points, layer = 3)

    return D



#TE_design_split_1
#3D_Split_1
#Please see the design split slide if you would like to change parameters
target_length = 15
duty_cycle = 0.59036
pitch = 0.664
radius = 30
y_span = 10
L_extra = 10
waveguide_width = 0.45
waveguide_length = 10

grating_coupler = make_grating_coupler(target_length, duty_cycle, pitch, radius, y_span, L_extra, waveguide_width, waveguide_length)

grating_coupler.add_port(name = 'output', midpoint = [-waveguide_length,0], width = waveguide_width, orientation = 180)
qp(grating_coupler)



grating_coupler.write_gds('Single_Si_grating_coupler_for_single_fiber.gds')
