# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""HTML hover layer for charts: crosshair + one tooltip listing every series."""

from __future__ import annotations

from html import escape

CHART_CSS = """
.ag-chart-wrap{position:relative;display:inline-block;max-width:100%}
.ag-chart-wrap svg{display:block;max-width:100%;height:auto}
.ag-ctip{position:absolute;pointer-events:none;background:var(--ag-surface,#fff);border:1px solid var(--ag-border,rgba(0,0,0,.1));
  border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.12);padding:8px 10px;font:12px/1.35 var(--ag-font,system-ui,sans-serif);
  color:var(--ag-ink,#0b0b0b);min-width:110px;z-index:5}
.ag-ctip[hidden]{display:none}
.ag-ctip .h{color:var(--ag-ink-3,#898781);margin-bottom:4px;font-variant-numeric:tabular-nums}
.ag-ctip .r{display:grid;grid-template-columns:14px auto 1fr;align-items:center;gap:6px}
.ag-ctip .r i{height:2.5px;border-radius:2px}
.ag-ctip .r b{font-weight:600;font-variant-numeric:tabular-nums}
.ag-ctip .r span{color:var(--ag-ink-2,#52514e)}
"""

CHART_JS = r"""
(function(){
  function fmt(v){ if(v===null||v===undefined) return "n/a"; if(Number.isInteger(v)) return v.toLocaleString();
    var a=Math.abs(v); if(a>=1000) return Math.round(v).toLocaleString(); return Number(v.toPrecision(4)).toString(); }
  function setup(wrap){
    var svg=wrap.querySelector("svg.ag-chart-svg"); if(!svg) return;
    var meta=svg.querySelector("metadata.ag-chart-data"); if(!meta) return;
    var d=JSON.parse(meta.textContent); if(d.type!=="line") return;
    var NS="http://www.w3.org/2000/svg", tip=document.createElement("div"); tip.className="ag-ctip"; tip.hidden=true; wrap.appendChild(tip);
    var hair=document.createElementNS(NS,"line"); hair.setAttribute("stroke","currentColor"); hair.setAttribute("stroke-width","1");
    hair.style.color=getComputedStyle(wrap).getPropertyValue("--ag-ink-3")||"#898781"; hair.setAttribute("visibility","hidden");
    hair.setAttribute("y1",d.plot[1]); hair.setAttribute("y2",d.plot[3]); svg.appendChild(hair);
    var dots=d.series.map(function(s){ var c=document.createElementNS(NS,"circle"); c.setAttribute("r","4"); c.setAttribute("fill",s.color);
      c.setAttribute("stroke",getComputedStyle(wrap).getPropertyValue("--ag-surface")||"#fff"); c.setAttribute("stroke-width","2");
      c.setAttribute("visibility","hidden"); svg.appendChild(c); return c; });
    function sx(v){ return d.plot[0]+(v-d.xd[0])/((d.xd[1]-d.xd[0])||1)*(d.plot[2]-d.plot[0]); }
    function sy(v){ return d.plot[3]-(v-d.yd[0])/((d.yd[1]-d.yd[0])||1)*(d.plot[3]-d.plot[1]); }
    function hide(){ tip.hidden=true; hair.setAttribute("visibility","hidden"); dots.forEach(function(c){c.setAttribute("visibility","hidden");}); }
    svg.addEventListener("pointermove",function(ev){
      var p=svg.createSVGPoint(); p.x=ev.clientX; p.y=ev.clientY; var q=p.matrixTransform(svg.getScreenCTM().inverse());
      if(q.x<d.plot[0]-8||q.x>d.plot[2]+8||q.y<d.plot[1]-8||q.y>d.plot[3]+8){ hide(); return; }
      var best=0, bd=Infinity; for(var i=0;i<d.x.length;i++){ var dd=Math.abs(sx(d.x[i])-q.x); if(dd<bd){bd=dd;best=i;} }
      var X=sx(d.x[best]); hair.setAttribute("x1",X); hair.setAttribute("x2",X); hair.setAttribute("visibility","visible");
      tip.textContent=""; var h=document.createElement("div"); h.className="h"; h.textContent=fmt(d.x[best]); tip.appendChild(h);
      var rows=d.series.map(function(s,k){ return {s:s,k:k,v:s.y[best]}; }).sort(function(a,b){ return (b.v||0)-(a.v||0); });
      rows.forEach(function(r){ var row=document.createElement("div"); row.className="r"; var i=document.createElement("i"); i.style.background=r.s.color;
        var b=document.createElement("b"); b.textContent=fmt(r.v); var sp=document.createElement("span"); sp.textContent=r.s.name;
        row.appendChild(i); row.appendChild(b); row.appendChild(sp); tip.appendChild(row);
        var c=dots[r.k]; if(r.v===null){ c.setAttribute("visibility","hidden"); } else { c.setAttribute("cx",X); c.setAttribute("cy",sy(r.v)); c.setAttribute("visibility","visible"); } });
      tip.hidden=false; var rect=wrap.getBoundingClientRect(), x=ev.clientX-rect.left+14, y=ev.clientY-rect.top+14;
      if(x+tip.offsetWidth>rect.width) x=ev.clientX-rect.left-tip.offsetWidth-14; tip.style.left=Math.max(0,x)+"px"; tip.style.top=Math.max(0,y)+"px";
    });
    svg.addEventListener("pointerleave",hide);
  }
  function boot(){ document.querySelectorAll(".ag-chart-wrap").forEach(function(w){ if(!w.agReady){ w.agReady=true; try{setup(w);}catch(e){console.error(e);} } }); }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",boot); else boot();
})();
"""


def chart_block(chart) -> str:
    """A chart wrapped for the hover runtime (no page wrapper)."""
    return f'<div class="ag-chart-wrap">{chart.svg}</div>'


def chart_page(chart, title: str) -> str:
    th = chart.theme
    vars_ = ";".join(f"{k}:{v}" for k, v in th.css_vars().items())
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title or 'chart')}</title><style>:root{{{vars_}}}html,body{{margin:0;background:{th.background}}}"
        f"body{{padding:16px;font-family:{th.font}}}{CHART_CSS}</style></head><body>"
        f"{chart_block(chart)}<script>{CHART_JS}</script></body></html>"
    )


__all__ = ["CHART_CSS", "CHART_JS", "chart_block", "chart_page"]
