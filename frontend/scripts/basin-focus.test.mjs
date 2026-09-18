import {test} from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {outsideMask,insideBasin} from "../src/lib/basin-focus.ts";

const ring=(x,y,size)=>[[x,y],[x+size,y],[x+size,y+size],[x,y+size],[x,y]];
const data={type:"FeatureCollection",features:[{type:"Feature",properties:{},geometry:{type:"MultiPolygon",coordinates:[[ring(10,10,5),ring(11,11,1)],[ring(20,20,3)]]}}]};
test("mask complements multipart polygons and their holes",()=>{
  const mask=outsideMask(data);
  for(const p of [[10.5,10.5],[11.5,11.5],[21,21],[18,18],[-50,-30]]) assert.notEqual(insideBasin(p,data),insideBasin(p,mask));
  assert.equal(outsideMask({type:"FeatureCollection",features:[]}).features.length,0);
});
test("BIG Welang has closed rings and correct regional footprint",()=>{
  const basin=JSON.parse(readFileSync(new URL("../public/geo/basin_boundary.geojson",import.meta.url),"utf8"));
  assert.equal(basin.features[0].properties.OBJECTID_1,11622);
  for(const r of basin.features[0].geometry.coordinates) assert.deepEqual(r[0],r.at(-1));
  assert.equal(insideBasin([112.86139,-7.65778],basin),true);
  assert.equal(insideBasin([113.02,-8.25],basin),false);
  const mask=outsideMask(basin);
  assert.equal(insideBasin([112.86139,-7.65778],mask),false);
  assert.equal(insideBasin([113.02,-8.25],mask),true);
  const stations=readFileSync(new URL("../src/lib/stations.ts",import.meta.url),"utf8");
  const outside=[...stations.matchAll(/name: "([^"]+)", latitude: (-?[\d.]+), longitude: (-?[\d.]+)/g)].filter(m=>!insideBasin([Number(m[3]),Number(m[2])],basin)).map(m=>m[1]);
  console.log("Stations outside BIG boundary:",outside);
});
