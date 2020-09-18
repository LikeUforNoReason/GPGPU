/**************************************************************************** 

Compile:
    gcc Assignment4.c -lGL -lGLU -lglut -o Assignment4

Run:
    vglrun ./Assignment4

****************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<GL/gl.h>
#include<GL/glu.h>
#include<GL/glut.h>


static GLfloat rotate = 0.0;

void init(void){

    GLfloat light_ambient[] = {0.0,0.0,1.0,1.0};
    GLfloat light_diffuse[] = {0.0,0.0,1.0,10.0};
    GLfloat light_specular[] = {1.0, 1.0, 10.0, 10.0};
    GLfloat light_position[] = {0.0, 0.0, 1.0, 0.0};
    GLfloat light_direction[] = {1.0,1.0,1.0,1.0};

    GLfloat mat_ambient[] = {0.0,0.0,1.0,10.0};
    GLfloat mat_diffuse[] = {0.8,0.8,0.8,5.0};
    GLfloat mat_specular[] = {0.5, 0.5, 0.5, 20.0};
    GLfloat high_shininess[] = {2.0};
    //GLfloat no_mat[] = {0.0, 0.0, 0.0, 1.0};

    glClearColor(0.0,0.0,0.0,0.0);
    glShadeModel(GL_SMOOTH);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,200);
    glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,120.0);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light_direction);

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, high_shininess);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);

}


void display(void){

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    glRotatef(rotate,0.0,1.0,0.0);
    glBindTexture(GL_TEXTURE_2D,1);
    glEnable(GL_TEXTURE_2D);
    GLUquadric *quad = gluNewQuadric();
    gluQuadricTexture(quad, 1);
    gluSphere(quad,5,20,20);
    glPopMatrix();
    glutSwapBuffers();
    glDisable(GL_TEXTURE_2D);

}

void reshape(int w, int h){

    glViewport(0,0,(GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    /*if(w <= h)
        glOrtho(-7.5, 7.5, -7.5*(GLfloat)h/(GLfloat)w, 7.5*(GLfloat)h/(GLfloat)w, -70.0, 70.0);
    else
    	glOrtho(-7.5*(GLfloat)w/(GLfloat)h, 7.5*(GLfloat)w/(GLfloat)h, -7.5, 7.5, -70.0, 70.0);
    */
    gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 0.1, 20);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt (0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void update(int value)
{

    rotate += 2.0f;
    if(rotate > 360.f)
    {
        rotate -= 360;
    }

    glutPostRedisplay();
    glutTimerFunc(25,update,0);

}

void load_texture(){

    FILE *fopen(), *fptr;
    char buf[512];
    int im_size, im_width, im_height, max_color;
    unsigned char *texture_bytes;
    char *parse;

    //Load a ppm file and hand it off to the graphics card.
    fptr = fopen("scuff.ppm","r");
    fgets(buf, 512, fptr);

    do
    {
        fgets(buf, 512, fptr);
    }while(buf[0]=='#');

    parse = strtok(buf," ");
    im_width = atoi(parse);

    parse = strtok(NULL,"\n");
    im_height = atoi(parse);

    fgets(buf,512, fptr);
    parse = strtok(buf," ");
    max_color = atoi(parse);

    im_size = im_width*im_height;
    texture_bytes = (unsigned char*)calloc(3, im_size);
    fread(texture_bytes,1,3*im_size,fptr);
    fclose(fptr);

    glBindTexture(GL_TEXTURE_2D,1);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,im_width,im_height,0,GL_RGB,GL_UNSIGNED_BYTE, texture_bytes);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    //we don't need the copy; the graphics card has its own row.
    cfree(texture_bytes);
}

void cleanup()
{
	//Release resources here.
	exit(0);
}

void getout(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 'q':
			cleanup();
		default:
			break;
	}
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow(argv[0]);
    init();
    glutDisplayFunc(display);
    glutKeyboardFunc(getout);
    load_texture();
    glutReshapeFunc(reshape);
    glutTimerFunc(25,update,0);
    glutMainLoop();
    return 0;
}



