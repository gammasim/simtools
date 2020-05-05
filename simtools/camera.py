from simtools.model.model_parameters import TWO_MIRROR_TELS, CAMERA_ROTATE_ANGLE


__all__ = ['Camera']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Camera:

    def readPixelList(self, cameraConfigFile):

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

    def plotPixelLayout(self, telescopeType, pixels, cameraInSkyCoor=False):

        fig, ax = plt.subplots()

        if telescopeType not in TWO_MIRROR_TELS:
            if not cameraInSkyCoor:
                pixels['y'] = [(-1)*yVal for yVal in pixels['y']]

        originalRotateAngle = rotateAngle = pixels['rotateAngle']
        rotateAngle += np.deg2rad(CAMERA_ROTATE_ANGLE[telescopeType])
        if rotateAngle != 0:
            for i_pix, xNow, yNow in enumerate(zip(pixels['x'], pixels['y'])):
                pixels['x'][i_pix] = xNow*np.cos(rotateAngle) - yNow*np.sin(rotateAngle)
                pixels['y'][i_pix] = xNow*np.sin(rotateAngle) + yNow*np.cos(rotateAngle)

        # Find a list of neighbours for each pixel
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

        # TODO Reached here, before moving on, move the rotation to its own method.

        pixels, edgePixels, offPixels = list(), list(), list()
        edgePixelIndices = list()
        plt.gcf().set_size_inches(8, 8)

        pixOrientAngle = 0
        if pixels['funnelShape'] == 1 or pixels['funnelShape'] == 3:
            if pixels['funnelShape'] == 3:
                pixOrientAngle = 30
            if rotateAngle > 0:
                pixOrientAngle += np.rad2deg(rotateAngle)

        for i_pix, pixel in enumerate(xyPixPos):
            if pixels['funnelShape'] == 1 or pixels['funnelShape'] == 3:
                hexagon = mpatches.RegularPolygon((pixel[0], pixel[1]), numVertices=6,
                                                  radius=pixelDiameter/np.sqrt(3),
                                                  orientation=np.deg2rad(pixOrientAngle))
                if pixOn[i_pix]:
                    if len(neighbours[i_pix]) < 6:
                        edgePixelIndices.append(i_pix)
                        edgePixels.append(hexagon)
                    else:
                        pixels.append(hexagon)
                else:
                    offPixels.append(hexagon)
            elif pixels['funnelShape'] == 2:
                square = mpatches.Rectangle((pixel[0] - pixelDiameter/2.,
                                             pixel[1] - pixelDiameter/2.),
                                            width=pixelDiameter, height=pixelDiameter)
                if pixOn[i_pix]:
                    if len(neighbours[i_pix]) < 4:
                        edgePixelIndices.append(i_pix)
                        edgePixels.append(square)
                    else:
                        pixels.append(square)
                else:
                    offPixels.append(square)

            if self.printPixelID and pixID[i_pix] < 51:
                fontSize = 4
                if telNow == 'SCT':
                    fontSize = 2
                plt.text(pixel[0],
                         pixel[1],
                         pixID[i_pix],
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=fontSize)

        ax.add_collection(PatchCollection(pixels, facecolor='none',
                                          edgecolor='black', linewidth=0.2))
        ax.add_collection(PatchCollection(edgePixels,
                                          facecolor=mcolors.to_rgb('brown') + (0.5,),
                                          edgecolor=mcolors.to_rgb('black') + (1,),
                                          linewidth=0.2))
        ax.add_collection(PatchCollection(offPixels, facecolor='black',
                                          edgecolor='black', linewidth=0.2))

        legendObjects = [legH.pixelObject(), legH.edgePixelObject()]
        legendLabels = ['Pixel', 'Edge pixel']
        if (type(pixels[0]) == mlp.patches.RegularPolygon):
            legendHandlerMap = {legH.pixelObject: legH.hexPixelHandler(),
                                legH.edgePixelObject: legH.hexEdgePixelHandler(),
                                legH.offPixelObject: legH.hexOffPixelHandler()}
        elif (type(pixels[0]) == mlp.patches.Rectangle):
            legendHandlerMap = {legH.pixelObject: legH.squarePixelHandler(),
                                legH.edgePixelObject: legH.squareEdgePixelHandler(),
                                legH.offPixelObject: legH.squareOffPixelHandler()}
        if len(offPixels) > 0:
            legendObjects.append(legH.offPixelObject())
            legendLabels.append('Disabled pixel')

        xTitle = 'Horizontal scale [cm]'
        yTitle = 'Vertical scale [cm]'
        plt.axis('equal')
        plt.grid(True)
        ax.set_axisbelow(True)
        plt.axis([min(xyPixPos[:, 0]), max(xyPixPos[:, 0]),
                  min(xyPixPos[:, 1])*1.42, max(xyPixPos[:, 1])*1.42])
        plt.xlabel(xTitle, fontsize=18, labelpad=0)
        plt.ylabel(yTitle, fontsize=18, labelpad=0)
        ax.set_title('Pixels layout in {0:s} camera'.format(self.infoTels[telNow]['title']),
                     fontsize=15, y=1.02)
        plt.tick_params(axis='both', which='major', labelsize=15)

        self.plotAxesDef(telNow, plt, originalRotateAngle)
        ax.text(0.02, 0.02, 'For an observer facing the camera',
                transform=ax.transAxes, color='black', fontsize=12)

        focalLengthTelNow = 0
        if 'effectiveFocalLength' in self.infoTels[telNow]:
            focalLengthTelNow = float(self.infoTels[telNow]['effectiveFocalLength'])
        elif telNow == 'MST-FlashCam' or telNow == 'MST-NectarCam':
            focalLengthTelNow = float(self.infoTels['MST-optics']['effectiveFocalLength'])
        elif telNow == 'SST-Camera':
            focalLengthTelNow = float(self.infoTels['SST-Structure']['effectiveFocalLength'])
        if focalLengthTelNow > 0:
            self.fovs[telNow], rEdgeAvg = self.calcFOV(xyPixPos,
                                                       edgePixelIndices,
                                                       focalLengthTelNow)
            ax.text(0.02, 0.96, r'$f_{\mathrm{eff}}$ = ' +
                    '{0:.3f} cm'.format(focalLengthTelNow),
                    transform=ax.transAxes, color='black', fontsize=12)
            ax.text(0.02, 0.92, 'Avg. edge radius = {0:.3f} cm'.format(rEdgeAvg),
                    transform=ax.transAxes, color='black', fontsize=12)
            ax.text(0.02, 0.88, 'FoV = {0:.3f} deg'.format(self.fovs[telNow]),
                    transform=ax.transAxes, color='black', fontsize=12)

        plt.legend(legendObjects, legendLabels,
                   handler_map=legendHandlerMap,
                   prop={'size': 11}, loc='upper right')

        ax.set_aspect('equal', 'datalim')
        plt.tight_layout()
        plt.savefig('./figures/pixelLayout-' + self.infoTels[telNow]['name'] + '.pdf')