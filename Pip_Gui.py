import math
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import font

##############################################################
# Create the main window
root = tk.Tk()
root.title("Tabbed GUI Example")
root.geometry("1000x700")
##############################################################
large_font = font.Font(family="Helvetica", size=10, weight="bold")
######################## THE GUI #############################
# Create a Notebook widget (tab container)
notebook = ttk.Notebook(root)
# Create frames for each tab
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)
tab4 = ttk.Frame(notebook)
tab5 = ttk.Frame(notebook)
tab6 = ttk.Frame(notebook)
# Add the frames to the notebook (tabs)
notebook.add(tab1, text="Documentation")
notebook.add(tab2, text="Math Description")
notebook.add(tab3, text="Input Parameters")
notebook.add(tab4, text="Check the Results")
notebook.add(tab5, text="Optimum values for the velocity and launching angle")
notebook.add(tab6, text="Angular velocity")
# Pack the notebook (make it fill the main window)
notebook.pack(expand=1, fill="both")
# Add some content to each tab
label1 = tk.Label(tab1, text="Documentation", padx=10, pady=10 , font =("Arial", 10))
label1.pack()
label2 = tk.Label(tab2, text="Math Description", padx=10, pady=10 , font =("Arial", 10))
label2.pack()
label3 = tk.Label(tab3, text="Input the Parameters", padx=10, pady=10 , font =("Arial", 10))
label3.pack()
label4 = tk.Label(tab4, text="Check The Results", padx=10, pady=10 , font =("Arial", 10))
label4.pack()
label5 = tk.Label(tab5, text="The Optimum Velocity and Angle", padx=10, pady=10 , font =("Arial", 10))
label5.pack()
label6 = tk.Label(tab6, text="The Angular Acceleration", padx=10, pady=10 , font =("Arial", 10))
label6.pack()
################################ TAB 1 #########################
text1 = tk.Text(tab1, wrap='word', padx=10, pady=10)
text1.insert('1.0', "These project is to calculate the optimum angluar velocity and the "
                    "launching angle following the newoton projectile mothion ")
text1.pack(expand=1, fill='both')


######################Image Adjust###############
def resize_image(image_path, width, height):
    image = Image.open(image_path)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)
##################### Documentation Images ############################
# Load an image using PIL and display it in the first tab
image_path1 = "Projectile-Motion.png"  # Replace with your image path
photo1 = resize_image(image_path1, 700, 350)
# Create a label to display the image and place it in the tab 1
label1 = tk.Label(tab1, image=photo1)
label1.image1 = photo1  # Keep a reference to the image to avoid garbage collection
# Center the image in the tab
label1.place(relx=0.5, rely=0.1, anchor='n')
# Load an image using PIL and display it in the first tab
image_path_2 = "Arm-Motion.png"  # Replace with your image path
photo_2 = resize_image(image_path_2, 1000, 350)
# Create a label to display the image and place it in the tab 1
label2 = tk.Label(tab1, image=photo_2)
label2.image_2 = photo_2  # Keep a reference to the image to avoid garbage collection
# Center the image in the tab
label2.place(relx=0.5, rely=0.98, anchor='s')
##################### Documentation Images #####################
################################ TAB 1 end #####################
################################################################
################################ TAB 2 #########################
# Create a frame to hold LaTeX formulas in tab1
latex_frame = ttk.Frame(tab2)
latex_frame.pack(expand=1, fill='both')
# Function to create a matplotlib figure with LaTeX
fig, (ax11, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax12) = plt.subplots(12, 1)
ax11.text(0.000005, 0.000001, 'The important formulas are as following :', fontsize=10, va='top', ha='left')
ax11.axis('off')
ax1.text(0.005, 0.05, r'${  v_{o}^2 =\frac{g\Delta x^2}{Sin(2\theta)-2\Delta y Cos(\theta)^2} } $', fontsize=10,
         va='top', ha='left')
ax1.axis('off')
ax2.text(0.005, 0.1, r'$ { G=g\Delta x^2 , k= sin(2\theta)-2\Delta ycos(\theta)^2 }$', fontsize=10, va='top', ha='left')
ax3.text(0.005, 0.15, r'${ \frac{dv_{o}^2}{d\theta}  = 0 }$', fontsize=10, va='top', ha='left')
ax4.text(0.005, 0.2, r'$ { f= k\frac{dG}{d\theta}-G\frac{dK}{d\theta} }$', fontsize=10, va='top', ha='left')
ax5.text(0.005, 0.25, r'$ { f´(\theta)= k\frac{d^2G}{d\theta^2}-G\frac{d^2K}{d\theta^2} }$', fontsize=10, va='top',
         ha='left')
ax6.text(0.005, 0.3, ' For Using Newton Method : ', fontsize=10, va='top', ha='left')
ax7.text(0.005, 0.35, r'$ { \theta_{n+1}=\theta_n- \frac{f(\theta)}{f´(\theta)}} $', fontsize=10, va='top', ha='left')
ax8.text(0.005, 0.4, r'$ {G´(\theta)= -2g\Delta x Rcos(\theta)} $', fontsize=10, va='top', ha='left')
ax9.text(0.005, 0.45, r'$ {k´(\theta)= 2 g \Delta x \Delta x´(\theta)} $', fontsize=10, va='top', ha='left')
ax10.text(0.005, 0.5, r'$ {k´´(\theta)= 2 g \Delta x \Delta x´(\theta)} $', fontsize=10, va='top', ha='left')
ax12.text(0.005, 0.55, r'$ {G´´(\theta)= g R^2 (2 cos(\theta)^2-sin(2\theta) )} $', fontsize=10, va='top', ha='left')

ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')
ax7.axis('off')
ax8.axis('off')
ax9.axis('off')
ax10.axis('off')
ax12.axis('off')
# Embed the figure in the tkinter GUI
canvas = FigureCanvasTkAgg(fig, master=latex_frame)
canvas.draw()
canvas.get_tk_widget().pack(expand=1, fill='both')
################################ TAB 2 end #########################
xT = 0.1
yA = 0.1
R = 0.1
yT = 0.1
theta = 0.10
tol = 0.10
#############################End Tab math ###################################

###############################INPUT TAB #####################################
def retrieve_inputs():
    global xT, yA, R, yT, theta, tol
    xT = float(entry1.get())
    yA = float(entry2.get())
    R = float(entry3.get())
    yT = float(entry4.get())
    theta = float(entry5.get())
    tol = float(entry6.get())


label1 = tk.Label(tab3, text="xT:", font =("Arial", 10))
label1.pack(padx=10, pady=5, anchor='center')
entry1 = tk.Entry(tab3, font =("Arial", 10))
entry1.pack(padx=10, pady=5, fill='y')

label2 = tk.Label(tab3, text="yA:", font =("Arial", 10))
label2.pack(padx=10, pady=5, anchor='center')
entry2 = tk.Entry(tab3, font =("Arial", 10))
entry2.pack(padx=10, pady=5, fill='y')

label3 = tk.Label(tab3, text="R:", font =("Arial", 10))
label3.pack(padx=10, pady=5, anchor='center')
entry3 = tk.Entry(tab3, font =("Arial", 10))
entry3.pack(padx=10, pady=5, fill='y')

label4 = tk.Label(tab3, text="yT:", font =("Arial", 10))
label4.pack(padx=10, pady=5, anchor='center')
entry4 = tk.Entry(tab3, font =("Arial", 10))
entry4.pack(padx=10, pady=5, fill='y')

label5 = tk.Label(tab3, text="theta:", font =("Arial", 10))
label5.pack(padx=10, pady=5, anchor='center')
entry5 = tk.Entry(tab3, font =("Arial", 10))
entry5.pack(padx=10, pady=5, fill='y')

label6 = tk.Label(tab3, text="tol:", font =("Arial", 10))
label6.pack(padx=10, pady=5, anchor='center')
entry6 = tk.Entry(tab3, font =("Arial", 10))
entry6.pack(padx=10, pady=5, fill='y')

#################################################
################# BUTTONS #########################
# Button to retrieve inputs from Entry widgets
button = tk.Button(tab3, text="Submit", font=large_font, command=retrieve_inputs)
button.pack(padx=10, pady=10, anchor='center')


############################## The main functions ###############################
# The Input in SI units :
# This only inputs as example
# xT = 4.203000
# yA = 1.55
# R = 0.505
# yT = 0.2619
# theta = 10  # initial value for the newton algorism
# tol = 0.01  # for the newton algorism

################# Functions #########################
def Dx(xT, R, theta):
    x = xT - R * math.sin(math.radians(theta))
    return x


def dx(xT, R, theta):
    dx = -R * math.cos(math.radians(theta))
    return dx


def Dy(yT, yA, R, theta):
    y = yT - yA + R * math.cos(math.radians(theta))
    return y


def dy(yT, yA, R, theta):
    dy = -R * math.sin(math.radians(theta))
    return dy


def G(xT, yT, yA, R, theta):
    G = 9.81 * Dx(xT, R, theta) ** 2
    return G


def dG(xT, yT, yA, R, theta):
    dG = 2 * 9.81 * Dx(xT, R, theta) * dx(xT, R, theta)
    return dG


def ddG(xT, yT, yA, R, theta):
    ddG = (9.81 * R ** 2) * (2 * math.cos(math.radians(theta)) ** 2 - math.sin(math.radians(2 * theta)))
    return ddG


def k(xT, yT, yA, R, theta):
    k = math.sin(math.radians(2 * theta)) * Dx(xT, R, theta) - 2 * Dy(yT, yA, R, theta) * math.cos(
        math.radians(theta)) ** 2
    return k


def dk(xT, yT, yA, R, theta):
    dk = 2 * Dx(xT, R, theta) * math.cos(math.radians(2 * theta)) + 2 * Dy(yT, yA, R, theta) * math.sin(
        math.radians(2 * theta))
    return dk


def ddk(xT, yT, yA, R, theta):
    ddk = 4 * Dy(yT, yA, R, theta) * math.cos(math.radians(2 * theta)) - 4 * Dx(xT, R, theta) * math.sin(
        math.radians(2 * theta)) - 2 * R * math.cos(math.radians(2 * theta))
    return ddk


def f(xT, yT, yA, R, theta):
    f = k(xT, yT, yA, R, theta) * dG(xT, yT, yA, R, theta) - G(xT, yT, yA, R, theta) * dk(xT, yT, yA, R, theta)
    return f


def df(xT, yT, yA, R, theta):
    df = k(xT, yT, yA, R, theta) * ddG(xT, yT, yA, R, theta) - G(xT, yT, yA, R, theta) * ddk(xT, yT, yA, R, theta)
    return df


def Newton_Theta(xT, yT, yA, R, theta, tol):
    while abs(f(xT, yT, yA, R, theta)) > tol:
        theta = theta - f(xT, yT, yA, R, theta) / df(xT, yT, yA, R, theta)
    return theta

opt_angle = 0.0
opt_velo = 0.0
def clear_field():
   global opt_angle
   global opt_velo
   opt_angle = 0.0
   opt_velo = 0.0
   result_angle_velo.delete(1.0, "end")

def opt_angle_velo():
    global opt_angle, opt_velo
    try:

        opt_angle = Newton_Theta(xT, yT, yA, R, theta, tol)
        opt_velo = np.sqrt(G(xT, yT, yA, R, opt_angle) / k(xT, yT, yA, R, opt_angle))
        result_angle_velo.delete(1.0, "end")
        result_angle_velo.insert(1.0,   f"{'The optimum angle  =' }{opt_angle} \n")
        result_angle_velo.insert(1.0,   f" { 'The optimum vlocity =' }{opt_velo}\n")

    except:
        clear_field()
        result_angle_velo.insert(1.0, "Error")



#
def fY(dX, opt_velo, opt_angle):
    fy = math.tan(math.radians(opt_angle)) * dX - (9.81 * dX ** 2) / (
            2 * (math.cos(math.radians(opt_angle)) ** 2) * opt_velo ** 2)
    return fy

def vel_t(dX, opt_velo, opt_angle, yA, R):
    global T, A_vel, t, ang_acce
    Y_1 = fY(dX, opt_velo, opt_angle)
    Y = Y_1 + (yA - R * math.cos(math.radians(opt_angle)))
    t = 2 * math.radians(opt_angle) / (opt_velo / R)
    ang_acce = ((opt_velo / R) ** 2) / (2 * math.radians(opt_angle))
    ang_velo = ang_acce * t
    T = []
    A_vel = []
    T.append(0)
    T.append(t)
    A_vel.append(0)
    A_vel.append(ang_velo)
    return T, A_vel
################# End Functions #########################
###########################Print results#################
#####################Optimum Motion Parameter ##############################
def Opt_Par():

        angles=np.arange(10,80,1)
        velocity=[]
        for i in range (len(angles)):
            velocity.append(np.sqrt ( G(xT,yT,yA,R,angles[i])/k(xT,yT,yA,R,angles[i] ) ))
        #plt.rcParams["figure.figsize"] = (10,5)
        fig_opt = plt.figure(figsize=(10, 5))
        ax_opt = fig_opt.add_subplot(111)
        ax_opt.plot(angles,velocity,'bo-',label= '   Motion Parameters  ')
        ax_opt.plot(opt_angle,opt_velo,'ks',label= '   Optimum Motion Parameters ')
        ax_opt.annotate([opt_angle,opt_velo], xy=(opt_angle,opt_velo), xytext=(opt_angle-1,opt_velo+2),arrowprops=dict(facecolor='black', shrink=0.001))
        ax_opt.set_xlabel('angle, degree')
        ax_opt.set_ylabel('velocity, m/s' )
        ax_opt.set_title("Study the Optimum conditions for the velocity and launching angle ")
        ax_opt.legend()
        #plt.show()
        fig_opt.savefig('Optimum conditions.png', format='png')

def Opt_Par_plot():
    ###################insert###############################
    global label_opt
    image_path_opt = "Optimum conditions.png"  # Replace with your image path
    photo_opt = resize_image(image_path_opt, 800, 600)
    # Create a label to display the image and place it in the tab 1
    label_opt = tk.Label(tab5, image=photo_opt)
    label_opt.image_opt = photo_opt # Keep a reference to the image to avoid garbage collection
    # Center the image in the tab
    label_opt.place(relx=0.5, rely=0.99 , anchor='s')

def clear_field_Opt_Par():
    global label_opt
    label_opt.destroy()
#####################End Optimum Motion Parameter##################

#########################Check The Results###########################################

def check_Reuslts():
        dX = np.arange(start=0, stop=xT + 0.1 * xT, step=0.1)
        X = dX + R * math.sin(math.radians(opt_angle))
        Y_1 = fY(dX, opt_velo, opt_angle)
        Y = Y_1 + (yA - R * math.cos(math.radians(opt_angle)))
        fig_check_Reuslts = plt.figure(figsize=(10, 5))
        ax_check_Reuslts = fig_check_Reuslts.add_subplot(111)
        ax_check_Reuslts.plot(X, Y, 'bo-', label='  Trajectory')
        ax_check_Reuslts.plot(xT, yT, 'ks', label='   Target ')
        ax_check_Reuslts.annotate([xT, yT], xy=(xT, yT), xytext=(xT - 1, yT - 1), arrowprops=dict(facecolor='black', shrink=0.001))
        ax_check_Reuslts.set_xlabel('x, m')
        ax_check_Reuslts.set_ylabel('y, m')
        ax_check_Reuslts.set_title("Check the Results")
        ax_check_Reuslts.legend()
        fig_check_Reuslts.savefig('check_Reuslts.png', format='png')

def check_Reuslts_plot():
    ###################insert###############################
    global label_check_Reuslts
    image_check_Reuslts = "check_Reuslts.png"  # Replace with your image path
    photo_check_Reuslts = resize_image(image_check_Reuslts, 800, 600)
    # Create a label to display the image and place it in the tab 1
    label_check_Reuslts = tk.Label(tab4, image=photo_check_Reuslts)
    label_check_Reuslts.image_results = photo_check_Reuslts  # Keep a reference to the image to avoid garbage collection
    # Center the image in the tab
    label_check_Reuslts.place(relx=0.5, rely=0.99, anchor='s')

def clear_field_check_Reuslts():
    global label_check_Reuslts
    label_check_Reuslts.destroy()


####################################################################







##############################Angular Velocity ######################################

def Ang_vel():
        dX = np.arange(start=0, stop=xT + 0.1 * xT, step=0.1)
        X = dX + R * math.sin(math.radians(opt_angle))
        Y_1 = fY(dX, opt_velo, opt_angle)
        Y = Y_1 + (yA - R * math.cos(math.radians(opt_angle)))

        t = 2 * math.radians(opt_angle) / (opt_velo / R)
        ang_acce = ((opt_velo / R) ** 2) / (2 * math.radians(opt_angle))
        ang_velo = ang_acce * t
        T = []
        A_vel = []
        T.append(0)
        T.append(t)
        A_vel.append(0)
        A_vel.append(ang_velo)


        fig_Ang_vel = plt.figure(figsize=(10, 5))
        ax_Ang_vel = fig_Ang_vel.add_subplot(111)
        ax_Ang_vel.plot(T, A_vel, 'bo-', label='  Angular velocity of the Arm ')
        ax_Ang_vel .annotate(["open the hand", "Time :", t, "Angular velocity :", ang_velo], xy=(t, ang_velo),
                     xytext=(0, ang_velo - 2), arrowprops=dict(facecolor='black', shrink=0.001))
        ax_Ang_vel.set_xlabel('Time,sec')
        ax_Ang_vel.set_ylabel('Angular velocity, rad/sec')
        ax_Ang_vel.set_title("The Arm motion")
        ax_Ang_vel.legend()
        fig_Ang_vel.savefig('Angular Velocity.png', format='png')


def Ang_vel_plot():
    ###################insert###############################
    global label_Ang_vel
    image_Ang_vel = "Angular Velocity.png"  # Replace with your image path
    photo_Ang_vel = resize_image(image_Ang_vel, 800, 600)
    # Create a label to display the image and place it in the tab 1
    label_Ang_vel = tk.Label(tab6, image=photo_Ang_vel)
    label_Ang_vel.image_Ang_vel= photo_Ang_vel  # Keep a reference to the image to avoid garbage collection
    # Center the image in the tab
    label_Ang_vel.place(relx=0.5, rely=0.99, anchor='s')


def clear_field_Ang_vel():
    global label_Ang_vel
    label_Ang_vel.destroy()

##############################Angular Velocity ######################################

####################################################################
#############################Calculation Buttons####################
button_calc = tk.Button(tab3, text="Calculate Operation ", font=large_font, command=opt_angle_velo)
button_calc.pack(padx=10, pady=10, anchor='center')
button_clear = tk.Button(tab3, text="Clear calculation ", font=large_font, command = clear_field)
button_clear.pack(padx=10, pady=10, anchor='center')
result_angle_velo = tk.Text(tab3, height=50, width=100, font =("Arial", 20))
result_angle_velo.pack(padx=10, pady=10, anchor='s')
####################################################################
###################Second Plot #####################################
button_Opt_Par = tk.Button(tab5, text=" Plotting Optimum Operation   ", font=large_font, command=Opt_Par)
button_Opt_Par.pack(padx=10, pady=10, anchor='center')

button_Opt_Par_plot = tk.Button(tab5, text=" Insert PLotting Optimum Operation   ", font=large_font, command=Opt_Par_plot)
button_Opt_Par_plot.pack(padx=10, pady=10, anchor='center')

button_Opt_Par_plot_clear = tk.Button(tab5, text=" Remove PLotting Optimum Operation   ", font=large_font, command=clear_field_Opt_Par)
button_Opt_Par_plot_clear.pack(padx=10, pady=10, anchor='center')
###################################################################
###################first plot####################################
button_check_Reuslts = tk.Button(tab4, text=" Check the Results  ", font=large_font, command=check_Reuslts)
button_check_Reuslts.pack(padx=10, pady=10, anchor='center')

button_check_Reuslts_plot = tk.Button(tab4, text=" Insert PLotting for Checking the Results   ", font=large_font, command=check_Reuslts_plot)
button_check_Reuslts_plot.pack(padx=10, pady=10, anchor='center')

button_check_Reuslts_plot_clear = tk.Button(tab4, text=" Remove PLotting for Checking the Results", font=large_font, command=clear_field_check_Reuslts)
button_check_Reuslts_plot_clear.pack(padx=10, pady=10, anchor='center')
##################################################################
####################Third Plot ####################################
button_Ang_vel = tk.Button(tab6, text=" Angular Velocity Calculation  ", font=large_font, command=Ang_vel)
button_Ang_vel.pack(padx=10, pady=10, anchor='center')

button_Ang_vel_plot = tk.Button(tab6, text=" Insert PLotting for Angular Velocity   ", font=large_font, command=Ang_vel_plot)
button_Ang_vel_plot.pack(padx=10, pady=10, anchor='center')

button_Ang_vel_plot_clear = tk.Button(tab6, text=" Remove PLotting for Angular Velocity", font=large_font, command=clear_field_Ang_vel)
button_Ang_vel_plot_clear.pack(padx=10, pady=10, anchor='center')


##################################################################

# Run the application
root.mainloop()


# Button to print important outputs
#button = tk.Button(tab3, text="Print", font=large_font, command=out)
#button.pack(padx=10, pady=10, anchor='center')

#def out():
    #print("The optimum angle = ", opt_angle)
    #print("The optimum velocity = ", opt_velo)
    #print("The Time of arm to reach optimum angle = ", t)
    #print("The  required moving angluar acceleration = ", ang_acce)
    #print("The optimum angle = ", opt_angle)