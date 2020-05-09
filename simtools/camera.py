import logging
import numpy as np
from scipy.spatial import cKDTree as KDTree
import matplotlib as mlp
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import colors as mcolors
from simtools.util import legendHandlers as legH
from simtools.model.telescope_model import TelescopeModel
from simtools.model.model_parameters import TWO_MIRROR_TELS, CAMERA_ROTATE_ANGLE

__all__ = ['Camera']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Camera:

    def readPixelList(self, cameraConfigFile):
        '''
        Read the pixel layout from the camera config file, assumed to be in a sim_telarray format.

        Parameters
        ----------
        cameraConfigFile: string
            The sim_telarray file name.

        Returns
        -------
        dict: pixels
            A dictionary with the pixel positions, the camera rotation angle,
            the pixel shape, the pixel diameter, the pixel IDs and their "on" status.

        Notes
        -----
        The pixel shape can be hexagonal (denoted as 1 or 3) or a square (denoted as 2).
        The hexagonal shapes differ in their orientation, where those denoted as 3 are rotated
        clockwise by 30 degrees with respect to those denoted as 1.
        '''

        datFile = open(cameraConfigFile, 'r')
        pixels = dict()
        pixels['diameter'] = 9999
        pixels['funnelShape'] = 9999
        pixels['rotateAngle'] = 0  # The LST and MST-NectarCam cameras need to be rotated
        pixels['x'] = list()
        pixels['y'] = list()
        pixels['pixID'] = list()
        pixels['pixOn'] = list()
        for line in datFile:
            if line.startswith('PixType'):
                pixels['funnelShape'] = int(line.split()[5].strip())
                pixels['diameter'] = float(line.split()[6].strip())
            if line.startswith('Rotate'):
                pixels['rotateAngle'] = np.deg2rad(float(line.split()[1].strip()))
            if line.startswith('Pixel'):
                pixInfo = line.split()
                pixels['x'].append(float(pixInfo[3].strip()))
                pixels['y'].append(float(pixInfo[4].strip()))
                pixels['pixID'].append(int(pixInfo[1].strip()))
                if len(pixInfo) > 9:
                    if int(pixInfo[9].strip()) != 0:
                        pixels['pixOn'].append(True)
                    else:
                        pixels['pixOn'].append(False)
                else:
                    pixels['pixOn'].append(True)

        if pixels['diameter'] == 9999:
            logger.warn('Could not read the pixel diameter from .dat file, '
                        'defaulting to 4.9 cm')
            pixels['diameter'] = 4.9

        return pixels

    def rotatePixels(self, telescopeType, pixels):
        '''
        Rotate the pixels according to the rotation angle given in pixels['rotateAngle'].
        Additional rotation is added to get to the camera view of an observer facing the camera.
        The angle for the axes rotation depends on the coordinate system in which the original
        data was provided.

        Parameters
        ----------
        telescopeType: string
            As provided by the telescope model method "TelescopeModel".
        pixels: dictionary
            The dictionary produced by the readPixelList method of this class

        Returns
        -------
        pixels: dict
            The pixels dictionary with rotated pixels.
            The pixels orientation for plotting is added to the dictionary in pixels['orientation'].
            The orientation is determined by the funnel shape (see readPixelList for details).

        Notes
        -----
        The additional rotation angle to get to the camera view of an observer facing the camera
        is saved in the const dictionary CAMERA_ROTATE_ANGLE.
        In the case of dual mirror telescopes, the axis is flipped in order to keep the same
        axis definition as for single mirror telescopes.
        The list of dual mirror telescopes is given in the const dictionary TWO_MIRROR_TELS.
        '''

        if telescopeType not in TWO_MIRROR_TELS:
            pixels['y'] = [(-1)*yVal for yVal in pixels['y']]

        rotateAngle = pixels['rotateAngle']  # So not to change the original angle
        rotateAngle += np.deg2rad(CAMERA_ROTATE_ANGLE[telescopeType])
        if rotateAngle != 0:
            for i_pix, xyPixPos in enumerate(zip(pixels['x'], pixels['y'])):
                pixels['x'][i_pix] = xyPixPos[0]*np.cos(rotateAngle) - xyPixPos[1]*np.sin(rotateAngle)
                pixels['y'][i_pix] = xyPixPos[0]*np.sin(rotateAngle) + xyPixPos[1]*np.cos(rotateAngle)

        pixels['orientation'] = 0
        if pixels['funnelShape'] == 1 or pixels['funnelShape'] == 3:
            if pixels['funnelShape'] == 3:
                pixels['orientation'] = 30
            if rotateAngle > 0:
                pixels['orientation'] += np.rad2deg(rotateAngle)

        return pixels

    def calcFOV(self, xPixel, yPixel, edgePixelIndices, focalLength):
        '''
        Calculate the FOV of the camera in degrees, taking into account the focal length.

        Parameters
        ----------
        xPixel: list
            List of positions of the pixels on the x-axis
        yPixel: list
            List of positions of the pixels on the y-axis
        edgePixelIndices: list
            List of indices of the edge pixels
        focalLength: float
            The focal length of the camera in (preferably the effective focal length)

        Returns
        -------
        fov: float
            The FOV of the camera in the degrees.
        averageEdgeDistance: float
            The average edge distance of the camera

        Notes
        -----
        The x,y pixel positions and focal length are assumed to have the same unit (usually cm)

        '''

        averageEdgeDistance = 0
        for i_pix in edgePixelIndices:
            averageEdgeDistance += np.sqrt(xPixel[i_pix] ** 2 + yPixel[i_pix] ** 2)
        averageEdgeDistance /= len(edgePixelIndices)

        fov = 2*np.rad2deg(np.arctan(averageEdgeDistance/focalLength))

        return fov, averageEdgeDistance

    def findNeighbours(self, xPos, yPos, radius):
        '''
        use a KD-Tree to quickly find nearest neighbours
        (e.g., of the pixels in a camera or mirror facets)

        Parameters
        ----------
        xPos : array_like
            x position of each e.g., pixel
        yPos : array_like
            y position of each e.g., pixel
        radius : float
            radius to consider neighbour it should be slightly larger
            than the pixel diameter or mirror facet.

        Returns
        -------
        neighbours: array_like
            Array of neighbour indices in a list for each e.g., pixel

        '''

        points = np.array([xPos, yPos]).T
        indices = np.arange(len(xPos))
        kdtree = KDTree(points)
        neighbours = [kdtree.query_ball_point(p, r=radius) for p in points]

        for neighbourNow, indexNow in zip(neighbours, indices):
            neighbourNow.remove(indexNow)  # get rid of the pixel or mirror itself

        return neighbours

    def findAdjacentNeighbourPixels(self, xPos, yPos, radius, rowColoumnDist):
        '''
        Find adjacent neighbour pixels in cameras with square pixels.
        Only directly adjacent neighbours are allowed, no diagonals.

        Parameters
        ----------
        xPos : array_like
            x position of each pixel
        yPos : array_like
            y position of each pixels
        radius : float
            radius to consider neighbour.
            Should be slightly larger than the pixel diameter.
        rowColoumnDist : float
            Maximum distance for pixels in the same row/column
            to consider when looking for a neighbour.
            Should be around 20% of the pixel diameter.

        Returns
        -------
        neighbours: array_like
            Array of neighbour indices in a list for each pixel

        '''

        # First find the neighbours with the usual method and the original radius
        # which does not allow for diagonal neighbours.
        neighbours = self.findNeighbours(xPos, yPos, radius)
        for i_pix, nn in enumerate(neighbours):
            # Find pixels defined as edge pixels now
            if len(nn) < 4:
                # Go over all other pixels and search for ones which are adjacent
                # but further than sqrt(2) away
                for j_pix in range(len(xPos)):
                    # No need to look at the pixel itself
                    # nor at any pixels already in the neighbours list
                    if j_pix != i_pix and j_pix not in nn:
                        dist = np.sqrt((xPos[i_pix] - xPos[j_pix])**2 +
                                       (yPos[i_pix] - yPos[j_pix])**2)
                        # Check if this pixel is in the same row or column
                        # and allow it to be ~1.68*diameter away (1.4*1.2 = 1.68)
                        # Need to increase the distance because of the curvature
                        # of the CHEC camera
                        if ((abs(xPos[i_pix] - xPos[j_pix]) < rowColoumnDist or
                            abs(yPos[i_pix] - yPos[j_pix]) < rowColoumnDist)
                                and dist < 1.2*radius):
                            nn.append(j_pix)

        return neighbours

    def getNeighbourPixels(self, pixels):
        '''
        Find adjacent neighbour pixels in cameras with hexagonal or square pixels.
        Only directly adjacent neighbours are searched for, no diagonals.

        Parameters
        ----------
        pixels: dictionary
            The dictionary produced by the readPixelList method of this class

        Returns
        -------
        neighbour: array_like
            Array of neighbour indices in a list for each pixel

        '''

        if pixels['funnelShape'] == 1 or pixels['funnelShape'] == 3:
            neighbours = self.findNeighbours(
                pixels['x'],
                pixels['y'],
                1.1*pixels['diameter']
            )
        elif pixels['funnelShape'] == 2:
            # Distance increased by 40% to take into account gaps in the SiPM cameras
            # Pixels in the same row/column can be 20% shifted from one another
            # Inside find_adjacent_neighbour_pixels the distance is increased
            # further for pixels in the same row/column to 1.68*diameter.
            neighbours = self.findAdjacentNeighbourPixels(
                pixels['x'],
                pixels['y'],
                1.4*pixels['diameter'],
                0.2*pixels['diameter']
            )

        return neighbours

    def plotAxesDef(self, telescopeType, plt, rotateAngle):
        '''
        Plot three axes definitions on the pyplot.plt instance provided.
        The three axes are Alt/Az, the camera coordinate system and
        the original coordinate system the pixel list was provided in.

        Parameters
        ----------
        telescopeType: string
            As provided by the telescope model method "TelescopeModel".
        plt: pyplot.plt instance
            A pyplot.plt instance where to add the axes definitions.
        rotateAngle: float
            The rotation angle applied

        '''

        invertYaxis = False
        xLeft = 0.7  # Position of the left most axis
        if telescopeType not in TWO_MIRROR_TELS:
            invertYaxis = True
            xLeft = 0.8

        xTitle = r'$x_{\!pix}$'
        yTitle = r'$y_{\!pix}$'
        xPos, yPos = (xLeft, 0.12)
        # The rotation of LST (above 100 degrees) raises the axes.
        # In this case, lower the starting point.
        if np.rad2deg(rotateAngle) > 100:
            yPos -= 0.09
            xPos -= 0.05
        axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'xPos': xPos, 'yPos': yPos,
                    'rotateAngle': rotateAngle - (1/2.)*np.pi, 'fc': 'black',
                    'ec': 'black', 'invertYaxis': invertYaxis}
        self.plotOneAxisDef(plt, **axesPars)

        xTitle = r'$x_{\!cam}$'
        yTitle = r'$y_{\!cam}$'
        xPos, yPos = (xLeft + 0.15, 0.12)
        axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'xPos': xPos, 'yPos': yPos,
                    'rotateAngle': (3/2.)*np.pi, 'fc': 'blue',
                    'ec': 'blue', 'invertYaxis': invertYaxis}
        self.plotOneAxisDef(plt, **axesPars)

        xTitle = 'Alt'
        yTitle = 'Az'
        xPos, yPos = (xLeft + 0.15, 0.25)
        axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'xPos': xPos, 'yPos': yPos,
                    'rotateAngle': (3/2.)*np.pi, 'fc': 'red',
                    'ec': 'red', 'invertYaxis': invertYaxis}
        self.plotOneAxisDef(plt, **axesPars)

        return

    def plotOneAxisDef(self, plt, **axesPars):
        '''
        Plot an axis on the pyplot.plt instance provided.

        Parameters
        ----------
        plt: pyplot.plt instance
            A pyplot.plt instance where to add the axes definitions.
        **axesPars: dict
             xTitle: str
                x-axis title
             yTitle: str
                y-axis title,
             xPos: float
                x position of the axis to draw
             yPos: float
                y position of the axis to draw
             rotateAngle: float
                rotation angle of the axis in radians
             fc: str
                face colour of the axis
             ec: str
                edge colour of the axis
             invertYaxis: bool
                Flag to invert the y-axis (for dual mirror telescopes).

        '''

        xTitle = axesPars['xTitle']
        yTitle = axesPars['yTitle']
        xPos, yPos = (axesPars['xPos'], axesPars['yPos'])

        r = 0.1  # size of arrow
        sign = 1.
        if axesPars['invertYaxis']:
            sign *= -1.
        xText1 = xPos + sign*r*np.cos(axesPars['rotateAngle'])
        yText1 = yPos + r*np.sin(0 + axesPars['rotateAngle'])
        xText2 = xPos + sign*r*np.cos(np.pi/2. + axesPars['rotateAngle'])
        yText2 = yPos + r*np.sin(np.pi/2. + axesPars['rotateAngle'])

        plt.gca().annotate(
            xTitle,
            xy=(xPos, yPos),
            xytext=(xText1, yText1),
            xycoords='axes fraction',
            ha='center',
            va='center',
            size='xx-large',
            arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0,
                            fc=axesPars['fc'], ec=axesPars['ec'])
        )

        plt.gca().annotate(
            yTitle,
            xy=(xPos, yPos),
            xytext=(xText2, yText2),
            xycoords='axes fraction',
            ha='center',
            va='center',
            size='xx-large',
            arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0,
                            fc=axesPars['fc'], ec=axesPars['ec'])
        )

        return

    def plotPixelLayout(self, telModel, pixels):
        '''
        Plot the pixel layout for an observer facing the camera.
        Including in the plot edge pixels, off pixels, pixel ID for the first 50 pixels,
        coordinate systems, FOV, focal length and the average edge radius.

        Parameters
        ----------
        telModel: an instance of TelescopeModel
            Assumed to be initialized to the requested telescope model.
        pixels: dictionary
            The dictionary produced by the readPixelList method of this class

        Returns
        -------
        plt: pyplot.plt instance with the pixel layout

        '''

        telescopeType = telModel.telescopeType

        pixels = self.rotatePixels(telescopeType, pixels)

        # Find a list of neighbours for each pixel
        neighbours = self.getNeighbourPixels(pixels)

        _, ax = plt.subplots()
        plt.gcf().set_size_inches(8, 8)

        onPixels, edgePixels, offPixels = list(), list(), list()
        # TODO move the "calculation" of edge pixels to its own method (less efficient, but looks better)
        edgePixelIndices = list()

        for i_pix, xyPixPos in enumerate(zip(pixels['x'], pixels['y'])):
            if pixels['funnelShape'] == 1 or pixels['funnelShape'] == 3:
                hexagon = mpatches.RegularPolygon(
                    (xyPixPos[0], xyPixPos[1]),
                    numVertices=6,
                    radius=pixels['diameter']/np.sqrt(3),
                    orientation=np.deg2rad(pixels['orientation'])
                )
                if pixels['pixOn'][i_pix]:
                    if len(neighbours[i_pix]) < 6:
                        edgePixelIndices.append(i_pix)
                        edgePixels.append(hexagon)
                    else:
                        onPixels.append(hexagon)
                else:
                    offPixels.append(hexagon)
            elif pixels['funnelShape'] == 2:
                square = mpatches.Rectangle(
                    (xyPixPos[0] - pixels['diameter']/2., xyPixPos[1] - pixels['diameter']/2.),
                    width=pixels['diameter'],
                    height=pixels['diameter']
                )
                if pixels['pixOn'][i_pix]:
                    if len(neighbours[i_pix]) < 4:
                        edgePixelIndices.append(i_pix)
                        edgePixels.append(square)
                    else:
                        onPixels.append(square)
                else:
                    offPixels.append(square)

            if pixels['pixID'][i_pix] < 51:
                fontSize = 4
                if telescopeType == 'SCT':
                    fontSize = 2
                plt.text(
                    xyPixPos[0],
                    xyPixPos[1],
                    pixels['pixID'][i_pix],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=fontSize
                )

        ax.add_collection(PatchCollection(
            onPixels, facecolor='none',
            edgecolor='black', linewidth=0.2)
        )
        ax.add_collection(PatchCollection(
            edgePixels,
            facecolor=mcolors.to_rgb('brown') + (0.5,),
            edgecolor=mcolors.to_rgb('black') + (1,),
            linewidth=0.2)
        )
        ax.add_collection(PatchCollection(
            offPixels, facecolor='black',
            edgecolor='black', linewidth=0.2)
        )

        legendObjects = [legH.pixelObject(), legH.edgePixelObject()]
        legendLabels = ['Pixel', 'Edge pixel']
        if (type(onPixels[0]) == mlp.patches.RegularPolygon):
            legendHandlerMap = {legH.pixelObject: legH.hexPixelHandler(),
                                legH.edgePixelObject: legH.hexEdgePixelHandler(),
                                legH.offPixelObject: legH.hexOffPixelHandler()}
        elif (type(onPixels[0]) == mlp.patches.Rectangle):
            legendHandlerMap = {legH.pixelObject: legH.squarePixelHandler(),
                                legH.edgePixelObject: legH.squareEdgePixelHandler(),
                                legH.offPixelObject: legH.squareOffPixelHandler()}
        if len(offPixels) > 0:
            legendObjects.append(legH.offPixelObject())
            legendLabels.append('Disabled pixel')

        plt.axis('equal')
        plt.grid(True)
        ax.set_axisbelow(True)
        plt.axis([min(pixels['x']), max(pixels['x']),
                  min(pixels['y'])*1.42, max(pixels['y'])*1.42])
        plt.xlabel('Horizontal scale [cm]', fontsize=18, labelpad=0)
        plt.ylabel('Vertical scale [cm]', fontsize=18, labelpad=0)
        ax.set_title('Pixels layout in {0:s} camera'.format(telescopeType),
                     fontsize=15, y=1.02)
        plt.tick_params(axis='both', which='major', labelsize=15)

        self.plotAxesDef(telescopeType, plt, pixels['rotateAngle'])
        ax.text(0.02, 0.02, 'For an observer facing the camera',
                transform=ax.transAxes, color='black', fontsize=12)

        focalLength = float(telModel.getParameter('effective_focal_length'))
        fov, rEdgeAvg = self.calcFOV(
            pixels['x'],
            pixels['y'],
            edgePixelIndices,
            focalLength
        )
        ax.text(
            0.02, 0.96,
            r'$f_{\mathrm{eff}}$ = ' + '{0:.3f} cm'.format(focalLength),
            transform=ax.transAxes,
            color='black',
            fontsize=12
        )
        ax.text(
            0.02, 0.92,
            'Avg. edge radius = {0:.3f} cm'.format(rEdgeAvg),
            transform=ax.transAxes,
            color='black',
            fontsize=12
        )
        ax.text(
            0.02, 0.88,
            'FoV = {0:.3f} deg'.format(fov),
            transform=ax.transAxes,
            color='black',
            fontsize=12
        )

        plt.legend(
            legendObjects,
            legendLabels,
            handler_map=legendHandlerMap,
            prop={'size': 11},
            loc='upper right'
        )

        ax.set_aspect('equal', 'datalim')
        plt.tight_layout()

        return plt
