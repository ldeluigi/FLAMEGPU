import xml.etree.ElementTree as ET
import sys, random, math, os
root = ET.Element('states')
it = ET.SubElement(root, 'itno')
it.text = '0'
env = ET.SubElement(root, 'environment')
if len(sys.argv) < 2:
    print("Missing argument: agent population size\n")
    exit(1)
for i in range(0, int(sys.argv[1])):
    agent = ET.SubElement(root, 'xagent')
    name = ET.SubElement(agent, 'name')
    name.text = 'turtle'
    x = ET.SubElement(agent, 'x')
    x.text = str(random.uniform(-10, 10))
    y = ET.SubElement(agent, 'y')
    y.text = str(random.uniform(-10, 10))
    heading = ET.SubElement(agent, 'heading')
    heading.text = str(random.uniform(0, 2 * math.pi))
tree = ET.ElementTree(root)
os.makedirs("../../iterations", exist_ok=True)
tree.write('../../iterations/0.xml', encoding='UTF-8')

