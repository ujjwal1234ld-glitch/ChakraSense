"""Local CAP test files only. No public transmission code."""
from datetime import datetime,timedelta,timezone
from uuid import uuid4
import xml.etree.ElementTree as ET
from .data import save_json

NS='urn:oasis:names:tc:emergency:cap:1.2'
ET.register_namespace('',NS)
def create_test_cap(storm,probability,approver,output):
    if not approver.strip():raise ValueError('A named reviewer must approve this TEST message.')
    if not 0<=probability<=1:raise ValueError('Probability outside [0,1].')
    now=datetime.now(timezone.utc).replace(microsecond=0)
    root=ET.Element('{'+NS+'}alert')
    def add(parent,key,text):return ET.SubElement(parent,'{'+NS+'}'+key).__setattr__('text',str(text))
    ident='chakrasense-test-'+str(uuid4())
    for k,v in [('identifier',ident),('sender','chakrasense.local.test'),('sent',now.isoformat()),('status','Test'),('msgType','Alert'),('scope','Private'),('addresses','local-demo-receiver'),('note','SOFTWARE TEST ONLY. Not an official weather warning.')]:add(root,k,v)
    info=ET.SubElement(root,'{'+NS+'}info')
    for k,v in [('language','en-IN'),('category','Met'),('event','ChakraSense research review test'),('urgency','Unknown'),('severity','Unknown'),('certainty','Unknown'),('expires',(now+timedelta(hours=1)).isoformat()),('senderName','ChakraSense local prototype'),('headline','TEST ONLY - '+str(storm)),('description',f'Artificial or research output. RI probability: {probability:.3f}. Not for public action.'),('instruction','Disregard as a public warning. Review software delivery only.')]:add(info,k,v)
    par=ET.SubElement(info,'{'+NS+'}parameter');add(par,'valueName','reviewer');add(par,'value',approver.strip())
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    save_json(str(output)+'.audit.json',{'id':ident,'approved_by':approver.strip(),'sent_at':now.isoformat(),'status':'Test','delivery':'local file only'})
    return root
