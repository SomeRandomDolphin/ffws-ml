import type { FeatureCollection, Polygon, MultiPolygon, Position } from "geojson";

export const emptyGeoJSON: FeatureCollection = {type:"FeatureCollection",features:[]};
export function basinPolygons(data: FeatureCollection): Position[][][] {
  return data.features.flatMap(feature => feature.geometry.type === "Polygon" ? [(feature.geometry as Polygon).coordinates] : feature.geometry.type === "MultiPolygon" ? (feature.geometry as MultiPolygon).coordinates : []);
}
function area(ring: Position[]) { return ring.slice(0,-1).reduce((sum,p,i)=>sum+p[0]*ring[i+1][1]-ring[i+1][0]*p[1],0); }
const orient = (ring: Position[], positive: boolean) => (area(ring)>0) === positive ? ring : [...ring].reverse();
export function outsideMask(data: FeatureCollection): FeatureCollection {
  const polygons=basinPolygons(data);
  if(!polygons.length) return emptyGeoJSON;
  const world: Position[]=[[-180,-85.0511],[180,-85.0511],[180,85.0511],[-180,85.0511],[-180,-85.0511]];
  return {type:"FeatureCollection",features:[
    {type:"Feature",properties:{},geometry:{type:"Polygon",coordinates:[world,...polygons.map(p=>orient(p[0],false))]}},
    ...polygons.flatMap(p=>p.slice(1).map(hole=>({type:"Feature" as const,properties:{},geometry:{type:"Polygon" as const,coordinates:[orient(hole,true)]}}))),
  ]};
}
function inRing(point: Position, ring: Position[]) {
  let inside=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++) {
    const a=ring[i],b=ring[j];
    const cross=(point[0]-a[0])*(b[1]-a[1])-(point[1]-a[1])*(b[0]-a[0]);
    if(Math.abs(cross)<1e-10 && point[0]>=Math.min(a[0],b[0]) && point[0]<=Math.max(a[0],b[0]) && point[1]>=Math.min(a[1],b[1]) && point[1]<=Math.max(a[1],b[1])) return true;
    if((a[1]>point[1])!==(b[1]>point[1]) && point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]) inside=!inside;
  }
  return inside;
}
export function insideBasin(point: Position,data: FeatureCollection) {
  return basinPolygons(data).some(p=>inRing(point,p[0]) && !p.slice(1).some(h=>inRing(point,h)));
}
