import type { StationGeo } from "./types";

export const stationGeography: StationGeo[] = [
  { name: "Bd. Suwoto", latitude: -7.841071, longitude: 112.698057, confidence: "Rendah", source: "user", description: "Titik pemantauan di hulu DAS Welang, dekat Jembatan Suwoto. Mencatat muka air di wilayah hulu sebelum aliran memasuki kawasan permukiman Pasuruan." },
  { name: "Krajan Timur", latitude: -7.817869, longitude: 112.732831, confidence: "Sedang-Tinggi", source: "user", description: "Stasiun pemantauan di sisi timur kawasan Krajan. Muka air di titik ini mencerminkan kondisi aliran yang melewati kawasan padat penduduk." },
  { name: "Purwodadi", latitude: -7.80376, longitude: 112.746, confidence: "Tinggi", source: "user", description: "Titik pemantauan di sekitar Purwodadi dengan keyakinan koordinat tinggi. Menjadi acuan kondisi muka air di segmen tengah Sungai Welang." },
  { name: "Bd. Sentono", latitude: -7.8425, longitude: 112.7896, confidence: "Sedang", source: "user", description: "Stasiun di jembatan Sentono yang melintasi Sungai Welang. Memantau variasi muka air pada segmen dengan catchmen hilir yang cukup luas." },
  { name: "Bd. Baong", latitude: -7.7747, longitude: 112.7737, confidence: "Sedang", source: "user", description: "Titik pemantauan di Jembatan Baong, hilir pertemuan aliran di bagian tengah DAS. Berguna melihat respons muka air terhadap hulu yang lebih luas." },
  { name: "Bd. Lecari", latitude: -7.7059, longitude: 112.7269, confidence: "Sedang", source: "user", description: "Stasiun di Jembatan Lecari pada segmen utara DAS. Muka air di titik ini dipengaruhi aliran dari wilayah hulu Prigen dan sekitarnya." },
  { name: "Bd. Bakalan", latitude: -7.748434, longitude: 112.759904, confidence: "Sedang-Rendah", source: "user", description: "Titik pemantauan di Jembatan Bakalan dengan keyakinan koordinat sedang-rendah. Mencatat kondisi aliran di segmen yang mengalir ke arah timur laut." },
  { name: "AWLR Kademungan", latitude: -7.77331, longitude: 112.78173, confidence: "Tinggi", source: "user", description: "Automatic Water Level Recorder di Kademungan dengan keyakinan koordinat tinggi. Memberikan rekaman muka air kontinu di segmen tengah Sungai Welang." },
  { name: "Bd. Domas", latitude: -7.721469, longitude: 112.812289, confidence: "Tinggi", source: "user", description: "Stasiun di Jembatan Domas, dekat hilir Sungai Welang bagian timur. Muka air di titik ini relevan untuk wilayah pertanian di sekitarnya." },
  { name: "Bd Guyangan", latitude: -7.658333, longitude: 112.822778, confidence: "Sedang-Tinggi", source: "user", description: "Titik pemantauan di Jembatan Guyangan, di wilayah yang dilayani oleh aliran Sungai Welang bagian hilir sebelum mendekati pesisir." },
  { name: "Bd. Grinting", latitude: -7.689107, longitude: 112.846016, confidence: "Tinggi", source: "user", description: "Stasiun di Jembatan Grinting dengan keyakinan koordinat tinggi. Memantau muka air di segmen hilir DAS Welang dekat kawasan Grinting." },
  { name: "Sidogiri", latitude: -7.668333, longitude: 112.831944, confidence: "Sedang", source: "user", description: "Titik pemantauan di kawasan Sidogiri, pesantren tertua di Pasuruan. Muka air di titik ini menjadi penting bagi wilayah sekitarnya yang padat aktivitas." },
  { name: "Klosod", latitude: -7.665417, longitude: 112.840278, confidence: "Sedang", source: "user", description: "Stasiun di Jembatan Klosod pada segmen hilir Sungai Welang. Memantau kondisi muka air sebelum aliran mendekati muara Dhompo." },
  { name: "Dhompo", latitude: -7.65778, longitude: 112.86139, confidence: "Tinggi", source: "user", description: "Stasiun utama di muara Dhompo, titik akhir Sungai Welang sebelum bermuara ke Laut Jawa. Muka air di titik ini dipengaruhi pasang surut laut dan menjadi indikator risiko rob serta banjir di pesisir Pasuruan." },
  { name: "Jalan Nasional", latitude: -7.629417, longitude: 112.874806, confidence: "Tinggi-Sedang", source: "user", description: "Titik pemantauan di sekitar jalur Jalan Nasional pesisir utara Jawa. Memantau muka air di titik yang strategis bagi akses transportasi lintas kabupaten." },
];

export function validateStationGeography(stations: StationGeo[]): StationGeo[] {
  if (!stations.length) throw new Error("Tidak ada metadata stasiun geografis.");
  for (const station of stations) {
    if (!station.name.trim()) throw new Error(`Nama stasiun kosong.`);
    if (station.latitude < -90 || station.latitude > 90) throw new Error(`Latitude tidak valid untuk ${station.name}.`);
    if (station.longitude < -180 || station.longitude > 180) throw new Error(`Longitude tidak valid untuk ${station.name}.`);
  }
  return stations;
}

validateStationGeography(stationGeography);
