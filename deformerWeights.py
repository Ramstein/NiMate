import maya.cmds as cmds
import maya.OpenMaya as om


from functools import partial
import socket
import sys
import os
import time
import threading
import math
import maya.utils as utils
import struct


def createRig(nsp='nsp1', res=5):
    # Create a plane with two clusters in a namespace.

    cmds.polyPlane(ch=1, w=10, h=10, sx=res, sy=res, ax=(0, 1, 0), name='%s:myShape' % nsp)
    cmds.cluster('%s:myShape' % nsp, name='%s:clusterA' % nsp)
    cmds.cluster('%s:myShape' % nsp, name='%s:clusterB' % nsp)


# --------------------------------------------------------------------
# Create the setup
# --------------------------------------------------------------------

# Clear file.
cmds.file(f=True, new=True)

# Create plane and two clusters.
createRig(nsp='nsp1')

# Modify some weights on clusterA.
cmds.select(['nsp1:myShape.vtx[6:11]'])
cmds.percent('nsp1:clusterA', v=0.5)

# Modify some weights on clusterB.
#
cmds.select(['nsp1:myShape.vtx[0:2]', 'nsp1:myShape.vtx[6:8]', 'nsp1:myShape.vtx[12:14]'])
cmds.percent('nsp1:clusterB', v=0.3)

# --------------------------------------------------------------------
# Export the weights in a variety of different ways
# --------------------------------------------------------------------

# Write cluster A weights.
#
cmds.deformerWeights('clusterA.xml', ex=True, deformer='nsp1:clusterA')

# Write cluster B weights, but do not write values of 1.0.
#
cmds.deformerWeights('clusterB.xml', ex=True, deformer='nsp1:clusterB', dv=1.0)

# Write cluster A and B weights at the same time.
#
cmds.deformerWeights('clusterAB.xml', ex=True, deformer=['nsp1:clusterA', 'nsp1:clusterB'])

# Export weights for all deformers on the shape, including vertex connections.
#
cmds.deformerWeights('shape_all.xml', ex=True, sh='nsp1:myShape', vc=True)

# Same as above skipping deformers matching '*B'.
#
cmds.deformerWeights('shape_NotB.xml', ex=True, sh='nsp1:myShape', vc=True, sk='*B')

# Export weights and attributes.
#
attributes = ['envelope', 'percentResolution', 'usePartialResolution']
cmds.deformerWeights('shape_all_attr.xml', ex=True, sh='nsp1:myShape', vc=True, at=attributes)

# Export name space nsp1: in scene to nsp2: in xml.
cmds.deformerWeights('shape_all_nsp2.xml', ex=True, sh='nsp1:myShape', vc=True, remap='nsp1:(.*);nsp2:$1')

# --------------------------------------------------------------------
# Import the weights
# --------------------------------------------------------------------

# Read both cluster's weight files separately.
cmds.deformerWeights('clusterA.xml', im=True, sh='nsp1:myShape', deformer='nsp1:clusterA')
cmds.deformerWeights('clusterB.xml', im=True, sh='nsp1:myShape', deformer='nsp1:clusterB')

# Read both deformers from the single file.
cmds.deformerWeights('shape_all.xml', im=True, sh='nsp1:myShape', deformer=['nsp1:clusterA', 'nsp1:clusterB'])

# Alternative way of reading both deformers.
cmds.deformerWeights('shape_all.xml', im=True, deformer=['nsp1:clusterA', 'nsp1:clusterB'])

# Read clusterA from the file containing both clusters.
cmds.deformerWeights('shape_all.xml', im=True, deformer='nsp1:clusterA')

#
# Create the same rig in a different namespace.
#
createRig(nsp='nsp2')

# Import weights from file that remapped the namespace on export.
cmds.deformerWeights('shape_all_nsp2.xml', im=True, sh='nsp2:myShape', deformer=['nsp2:clusterA', 'nsp2:clusterB'])

# Import weights from file containing a different namespace, and remap the namespace on import.
cmds.deformerWeights('shape_all.xml', im=True, sh='nsp2:myShape', deformer=['nsp2:clusterA', 'nsp2:clusterB'],
                     remap='nsp1:(.*);nsp2:$1')

#
# Create similar rig with different resolution (topology) in a different namespace.
#
createRig(nsp='nsp3', res=8)

# Import weights from file, remap the namespace on import, and use the barycentric method to remap the weight values.
cmds.deformerWeights('shape_all.xml', im=True, sh='nsp3:myShape', deformer=['nsp3:clusterA', 'nsp3:clusterB'],
                     remap='nsp1:(.*);nsp3:$1', method='barycentric')
