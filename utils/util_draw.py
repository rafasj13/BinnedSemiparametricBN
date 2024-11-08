
import pybnesian as pbn
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import os


import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pydot as dot

import pyAgrum.lib.bn2graph as ggr
import  pyAgrum.lib.utils as gutils
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def compare_graphs(model1, model2, filename="comparison_nets.pdf", size=10, noStyle=False):
    # Generate DAG for model1
    dag1 = ""
    for arc in model1.arcs():
        dag1 += f"{arc[0]}->{arc[1]};"

    bn1 = gum.fastBN(dag1)

    # Generate DAG for model2
    dag2 = ""
    for arc in model2.arcs():
        dag2 += f"{arc[0]}->{arc[1]};"

    bn2 = gum.fastBN(dag2)

    # Generate SVG
    svg = gnb.getGraph(graphDiff(bn1, bn2, model1, model2, False), size=size)
    with open("file.svg", "w") as f:
        f.write(svg)


    # Convert SVG to ReportLab Drawing
    drawing = svg2rlg("file.svg")
    if drawing is None:
        print("Error: Failed to convert SVG to ReportLab drawing.")
        return

    # Render PDF
    try:
        renderPDF.drawToFile(drawing, filename)
    except Exception as e:
        print(f"Error: Failed to render PDF. {e}")

    # Clean up
    os.remove("file.svg")



def plot_model(model, ax):

    DG = nx.DiGraph()
    DG.add_nodes_from(model.nodes())
    DG.add_edges_from(model.arcs())

    if isinstance(model, pbn.BayesianNetworkBase):
        for node in DG.nodes:
            if (model.node_type(node) == pbn.CKDEType()):
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'
            elif (model.node_type(node) == pbn.FBKernelType()):
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = '#D2D0FF'

    a = nx.nx_agraph.to_agraph(DG)
    a.write('graph.dot')
    a.clear()

    png_out = 'graph.png'

    # Use 'dot' command to convert DOT file to PNG
    subprocess.run(["dot", "-Tpng", "graph.dot", "-o", png_out])

    # Display the image using matplotlib (optional)
    img = plt.imread(png_out)
    plotimg = ax.imshow(img)
    ax.axis('off')

    os.remove(png_out)
    os.remove('graph.dot')
    return plotimg

def draw_model(model, filename, save=False):

    DG = nx.DiGraph()
    DG.add_nodes_from(model.nodes())
    DG.add_edges_from(model.arcs())
    

    if isinstance(model, pbn.BayesianNetworkBase):
        for node in DG.nodes:
            if (model.node_type(node) == pbn.CKDEType()):
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'
            elif (model.node_type(node) == pbn.FBKernelType()):
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = '#D2D0FF'

    a = nx.nx_agraph.to_agraph(DG)
    if filename[-4:] != '.dot':
        filename += '.dot'
    a.write(filename)
    a.clear()

    pdf_out = filename[:-4] + '.pdf'

    subprocess.run(["dot", "-Tpdf", filename, "-o", pdf_out])
    if save:
        model.save(filename[-4:], include_cpd=True)


def draw_model_pdag(model, filename):

    DG = nx.DiGraph()
    DG.add_nodes_from(model.nodes())
    DG.add_edges_from(model.arcs())

    # Add undirected edges
    UG = nx.Graph()
    UG.add_nodes_from(model.nodes())
    UG.add_edges_from(model.edges())

    if isinstance(model, pbn.BayesianNetworkBase):
        for node in DG.nodes:
            if model.node_type(node) == pbn.CKDEType():
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'
            elif model.node_type(node) == pbn.FBKernelType():
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = '#D2D0FF'

    # Convert to AGraph for visualization
    a = nx.nx_agraph.to_agraph(DG)

    # Add undirected edges to the AGraph
    for edge in UG.edges:
        a.add_edge(edge[0], edge[1], dir='none')  # 'dir=none' makes the edge undirected

    if filename[-4:] != '.dot':
        filename += '.dot'
    a.write(filename)
    a.clear()

    pdf_out = filename[:-4] + '.pdf'

    subprocess.run(["dot", "-Tpdf", filename, "-o", pdf_out])
    # model.save(filename[-4:], include_cpd=True)


def graphDiff(bnref, bncmp, bnref_pbn, bncmp_pbn, noStyle=False):
    """ Return a pydot graph that compares the arcs of bnref to bncmp.
    Includes coloring of nodes based on types and fills differently if types differ.
    """
    g = ggr.BN2dot(bnref)
    positions = gutils.dot_layout(g)

    res = dot.Dot(graph_type='digraph', bgcolor="transparent", layout="fdp", splines=True)

    for i1 in bnref.nodes():
        node_name = bnref.variable(i1).name()
        pos = positions[node_name]
        fill_color = "white"
        style = "filled"

        # Determine the type of the node in the reference network
        if isinstance(bnref_pbn, pbn.BayesianNetworkBase):
            if bnref_pbn.node_type(node_name) == pbn.CKDEType():
                fill_color = "gray"
            elif bnref_pbn.node_type(node_name) == pbn.FBKernelType():
                fill_color = "#D2D0FF"

        # Check if the node is present in the comparison network and determine style
        if node_name in bncmp.names():
            if isinstance(bncmp_pbn, pbn.BayesianNetworkBase):
                cmp_type = bncmp_pbn.node_type(node_name)
                ref_type = bnref_pbn.node_type(node_name)
                
                if cmp_type == pbn.CKDEType():
                    fill_color = "gray"
                elif cmp_type == pbn.FBKernelType():
                    fill_color = "#D2D0FF"
                # If types differ between bnref and bncmp, use a different pattern
                if cmp_type != ref_type:
                    if (ref_type == pbn.CKDEType() and cmp_type == pbn.FBKernelType()) or \
                        cmp_type == pbn.CKDEType() and ref_type == pbn.FBKernelType():
                            style = "filled"  
                    else:
                        style = "dashed"
       
                    

            # Add the node with the specified style and fill color
            res.add_node(dot.Node(f'"{node_name}"',
                                  style=style,
                                  fillcolor=fill_color,
                                  color="black",
                                  fontcolor="black"))
        else:
            # Node missing in bncmp; add with a dashed style if not noStyle
            if not noStyle:
                res.add_node(dot.Node(f'"{node_name}"',
                                      style="dashed",
                                      fillcolor=fill_color,
                                      color="black",
                                      fontcolor="black"
                                      ))

    # Handle arcs and the rest as before, adjusting as necessary for styles and positions
    if noStyle:
        for (i1, i2) in bncmp.arcs():
            n1 = bncmp.variable(i1).name()
            n2 = bncmp.variable(i2).name()
            res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"',
                                  style=gum.config["notebook", "graphdiff_correct_style"],
                                  color=gum.config["notebook", "graphdiff_correct_color"]))

    else:
        for (i1, i2) in bnref.arcs():
            n1 = bnref.variable(i1).name()
            n2 = bnref.variable(i2).name()

            # Check if the nodes exist in bncmp and if the arc should be present
            if not (n1 in bncmp.names() and n2 in bncmp.names()):
                res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"',
                                      style=gum.config["notebook", "graphdiff_missing_style"],
                                      color=gum.config["notebook", "graphdiff_missing_color"]))
                continue

            if bncmp.existsArc(n1, n2):  # Arc is OK in bncmp
                res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"',
                                      style=gum.config["notebook", "graphdiff_correct_style"],
                                      color="black"))
            elif bncmp.existsArc(n2, n1):  # Arc is reversed in bncmp
                res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"', style="invis"))
                res.add_edge(dot.Edge(f'"{n2}"', f'"{n1}"',
                                      style=gum.config["notebook", "graphdiff_reversed_style"],
                                      color=gum.config["notebook", "graphdiff_reversed_color"]))
            else:  # Arc is missing in bncmp
                res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"',
                                      style=gum.config["notebook", "graphdiff_missing_style"],
                                      color=gum.config["notebook", "graphdiff_missing_color"]))

        for (i1, i2) in bncmp.arcs():
            n1 = bncmp.variable(i1).name()
            n2 = bncmp.variable(i2).name()
            if not bnref.existsArc(n1, n2) and not bnref.existsArc(n2, n1):  # Arc only in bncmp
                res.add_edge(dot.Edge(f'"{n1}"', f'"{n2}"',
                                      style=gum.config["notebook", "graphdiff_overflow_style"],
                                      color=gum.config["notebook", "graphdiff_overflow_color"]))

    # Apply the layout with positions for consistency
    gutils.apply_dot_layout(res, positions)

    return res

def graphDiffLegend():
  try:
    # pydot is optional
    # pylint: disable=import-outside-toplevel
    import pydot as dot
  except ImportError:
    return None

  res = dot.Dot(graph_type='digraph', bgcolor="white", rankdir="LR")
  for i in "abcdefgh":
    res.add_node(dot.Node(i, style="invis"))
  res.add_edge(dot.Edge("a", "b", label="Overflow",
                        style=gum.config["notebook", "graphdiff_overflow_style"],
                        color=gum.config["notebook", "graphdiff_overflow_color"]))
  res.add_edge(dot.Edge("c", "d", label="Missing",
                        style=gum.config["notebook", "graphdiff_missing_style"],
                        color=gum.config["notebook", "graphdiff_missing_color"]))
  res.add_edge(dot.Edge("e", "f", label="Reversed",
                        style=gum.config["notebook", "graphdiff_reversed_style"],
                        color=gum.config["notebook", "graphdiff_reversed_color"]))
  res.add_edge(dot.Edge("g", "h", label="Correct",
                        style=gum.config["notebook", "graphdiff_correct_style"],
                        color="black"))

  # Add the node with the label on top
  res.add_node(dot.Node('A',
                        style="dashed",
                        fillcolor="white",
                        color="black",
                        fontcolor="black",
                        xlabel="Incorrect Node Type",
                        ))

    # Add invisible edges to center the node with the arrows
    
  res.add_edge(dot.Edge("b", "A", style="invis"))
  res.add_edge(dot.Edge("d", "A", style="invis"))
  res.add_edge(dot.Edge("f", "A", style="invis"))
  res.add_edge(dot.Edge("h", "A", style="invis"))

  return res