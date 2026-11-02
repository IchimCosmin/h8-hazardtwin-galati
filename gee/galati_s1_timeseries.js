// H8 HazardTwin — Galați: Sentinel-1 time series over the Danube floodplain.
// Run in the Earth Engine Code Editor (code.earthengine.google.com).
// Output: one CSV row per S1 acquisition with median VV/VH and the share of
// "new water" pixels (VV below threshold, outside permanent water) in the AOI.
// Status: written for CP1, NOT YET RUN (needs a GEE account) — planned for CP2.

var AOI = ee.Geometry.Rectangle([27.90, 45.33, 28.25, 45.53]);   // Galați + Siret/Prut confluence, lon/lat
var START = '2015-01-01';
var END   = '2026-11-01';
var WATER_DB = -18;            // VV threshold (dB) for open water; tuned later with Otsu
var PERMANENT_OCC = 80;        // JRC GSW occurrence (%) treated as permanent water

var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0);
var notPermanent = gsw.lt(PERMANENT_OCC);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI)
  .filterDate(START, END)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

var perImage = s1.map(function (img) {
  var vv = img.select('VV');
  var newWater = vv.lt(WATER_DB).and(notPermanent).rename('new_water');
  var valid = vv.mask().rename('valid');
  var stats = img.reduceRegion({
    reducer: ee.Reducer.median(), geometry: AOI, scale: 20, maxPixels: 1e10, bestEffort: true
  });
  var waterFrac = newWater.updateMask(notPermanent).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: AOI, scale: 20, maxPixels: 1e10, bestEffort: true
  }).get('new_water');
  var cover = valid.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: AOI, scale: 100, maxPixels: 1e10, bestEffort: true
  }).get('valid');
  return ee.Feature(null, {
    date: img.date().format('YYYY-MM-dd'),
    time_utc: img.date().format('HH:mm'),
    platform: img.get('platform_number'),
    orbit: img.get('orbitProperties_pass'),
    rel_orbit: img.get('relativeOrbitNumber_start'),
    vv_median_db: stats.get('VV'),
    vh_median_db: stats.get('VH'),
    new_water_frac: waterFrac,
    aoi_coverage: cover
  });
});

print('S1 images in AOI:', s1.size());
print('First rows:', perImage.limit(5));
Map.centerObject(AOI, 11);
Map.addLayer(AOI, {color: 'orange'}, 'AOI');
Map.addLayer(s1.filterDate('2024-09-14', '2024-09-30').first().select('VV'),
             {min: -25, max: 0}, 'S1 VV after EMSR758 (Sept 2024)');

Export.table.toDrive({
  collection: perImage,
  description: 'H8_galati_s1_timeseries',
  fileFormat: 'CSV',
  selectors: ['date', 'time_utc', 'platform', 'orbit', 'rel_orbit',
              'vv_median_db', 'vh_median_db', 'new_water_frac', 'aoi_coverage']
});
