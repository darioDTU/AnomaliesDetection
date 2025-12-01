import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

interface MapProps {
  latitude: number;
  longitude: number;
  rectangleBounds: [number, number][];
  onMapClick?: (lng: number, lat: number) => void;
}

export default function Map({ latitude, longitude, rectangleBounds , onMapClick}: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!mapContainer.current) {
      return;
    }

    let map = mapRef.current;

    if (!map) {
      map = new maplibregl.Map({
        container: mapContainer.current,
        style: 'https://api.maptiler.com/maps/019adaca-9392-7610-a1e0-89c3cf4bb095/style.json?key=ZaJJlmg85RpSs3eUudEF',
        center: [longitude, latitude],
        zoom: 4,
      });
      mapRef.current = map;

      map.on('click', (e) => {
        // Shift + left-click (button === 0)
        if (e.originalEvent.shiftKey && e.originalEvent.button === 0) {
          e.preventDefault();
          if (onMapClick) {
            const { lng, lat } = e.lngLat;
            onMapClick(lng, lat);
            console.log('Map clicked:', lng, lat);
          }
        }
      });
    }

    // Wait for style to load before adding sources/layers
    function addRectangle() {
      // Remove previous layers/sources
      if (map && map.getLayer('rectangle-layer')) {
        map.removeLayer('rectangle-layer');
      }
      if (map && map.getLayer('rectangle-outline')) {
        map.removeLayer('rectangle-outline');
      }
      if (map && map.getSource('rectangle-source')) {
        map.removeSource('rectangle-source');
      }

      // Add rectangle as GeoJSON source
      if (map) {
        map.addSource('rectangle-source', {
          type: 'geojson',
          data: {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [rectangleBounds],
            },
          },
        });
      }

      // Fill layer
      if (map) {
        map.addLayer({
          id: 'rectangle-layer',
          type: 'fill',
          source: 'rectangle-source',
          paint: {
            'fill-color': '#088',
            'fill-opacity': 0.8,
          },
        });

        // Outline layer
        map.addLayer({
          id: 'rectangle-outline',
          type: 'line',
          source: 'rectangle-source',
          paint: {
            'line-color': '#055',
            'line-width': 2,
          },
        });

        map.setCenter([longitude, latitude]);
      }
    }

    if (map.isStyleLoaded()) {
      addRectangle();
    } else {
      map.once('load', addRectangle);
    }

    // Cleanup on unmount
    return () => {
      if (map.getLayer('rectangle-layer')) map.removeLayer('rectangle-layer');
      if (map.getLayer('rectangle-outline')) map.removeLayer('rectangle-outline');
      if (map.getSource('rectangle-source')) map.removeSource('rectangle-source');
    };
  }, [latitude, longitude, rectangleBounds, onMapClick]);

  return <div ref={mapContainer} style={{ width: '100%', height: '100%'}} />;
}