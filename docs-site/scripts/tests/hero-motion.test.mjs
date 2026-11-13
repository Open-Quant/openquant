import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../../src/components/ResearchHero.astro',import.meta.url),'utf8').split('<script>')[1].split('</script>')[0];
class Target {
  constructor(){this.handlers={};this.attrs={};this.style={};this.dataset={};}
  addEventListener(k,fn){(this.handlers[k]??=new Set()).add(fn);}
  removeEventListener(k,fn){this.handlers[k]?.delete(fn);}
  fire(k,e={}){for(const fn of this.handlers[k]??[])fn(e);}
  setAttribute(k,v){this.attrs[k]=v;}
  toggleAttribute(k,v){if(v)this.attrs[k]='';else delete this.attrs[k];}
  getBoundingClientRect(){return {left:0,width:440};}
  setPointerCapture(id){this.capture=id;}
  hasPointerCapture(id){return this.capture===id;}
  releasePointerCapture(){this.capture=null;}
}
function setup(reduced=false){
  const nodes=Object.fromEntries(['.oq-plot','.oq-motion','#oq-reveal rect','.oq-scan'].map(k=>[k,new Target()]));
  const labels=[.12,.44,.76].map(n=>{const t=new Target();t.dataset.at=String(n);return t;});
  const media=new Target();media.matches=reduced;const doc=new Target();doc.hidden=false;
  let C, seq=0;const frames=new Map();let observer;
  class Host extends Target{querySelector(k){return nodes[k];}querySelectorAll(){return labels;}}
  vm.runInNewContext(source,{HTMLElement:Host,customElements:{get:()=>false,define:(_,c)=>C=c},matchMedia:()=>media,document:doc,requestAnimationFrame:fn=>{frames.set(++seq,fn);return seq;},cancelAnimationFrame:id=>frames.delete(id),IntersectionObserver:class{constructor(fn){observer=fn;}observe(){}disconnect(){}}});
  const host=new C();host.connectedCallback();
  const step=(time)=>{const pending=[...frames.values()];frames.clear();pending.forEach(fn=>fn(time));};
  return {host,nodes,media,doc,frames,step,visible:v=>observer([{isIntersecting:v}])};
}
test('autoplay reveals layers and can pause without a bottom control',()=>{const x=setup();assert.equal(x.nodes['#oq-reveal rect'].attrs.width,'0');x.step(100);x.step(200);assert.ok(+x.nodes['#oq-reveal rect'].attrs.width>0);x.nodes['.oq-motion'].fire('click');assert.equal(x.frames.size,0);assert.equal(x.nodes['.oq-motion'].attrs['aria-label'],'Resume animation');});
test('pointer and keyboard explore; leaving resumes',()=>{const x=setup();const p=x.nodes['.oq-plot'];p.fire('pointermove',{clientX:220});assert.equal(p.attrs['aria-valuenow'],'50');assert.equal(x.frames.size,0);p.fire('pointerleave');assert.equal(x.frames.size,1);p.fire('keydown',{key:'Home',preventDefault(){}});assert.equal(p.attrs['aria-valuenow'],'0');p.fire('keydown',{key:'End',preventDefault(){}});assert.equal(p.attrs['aria-valuenow'],'100');});
test('reduced motion stays static but permits explicit exploration',()=>{const x=setup(true);assert.equal(x.frames.size,0);assert.equal(x.nodes['.oq-plot'].attrs['aria-valuenow'],'100');assert.equal(x.nodes['.oq-motion'].hidden,true);x.nodes['.oq-plot'].fire('pointermove',{clientX:28});assert.equal(x.nodes['.oq-plot'].attrs['aria-valuenow'],'0');});
test('visibility and disconnect stop work; reconnect does not duplicate handlers',()=>{const x=setup();x.visible(false);assert.equal(x.frames.size,0);x.visible(true);assert.equal(x.frames.size,1);x.doc.hidden=true;x.doc.fire('visibilitychange');assert.equal(x.frames.size,0);x.host.disconnectedCallback();assert.equal(x.nodes['.oq-motion'].handlers.click.size,0);x.doc.hidden=false;x.host.connectedCallback();assert.equal(x.nodes['.oq-motion'].handlers.click.size,1);});
