import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

type FloatingSidebarProps = {
    latitude: number;
    longitude: number;
    setLatitude: Dispatch<SetStateAction<number>>;
    setLongitude: Dispatch<SetStateAction<number>>;
}

const FloatingSidebar = ({
    latitude,
    longitude,
    setLatitude,
    setLongitude
}: FloatingSidebarProps) => {
    const [latitudeInput, setLatitudeInput] = useState<string>(String(latitude));
    const [longitudeInput, setLongitudeInput] = useState<string>(String(longitude));

    useEffect(() => {
        setLatitudeInput(String(latitude));
    }, [latitude]);

    useEffect(() => {
        setLongitudeInput(String(longitude));
    }, [longitude]);

    const commitCoordinateIfValid = (raw: string, set : Dispatch<SetStateAction<number>>) => {
        const trimmed = raw.trim();
        if (trimmed === "" || trimmed === "-" || trimmed === "+") return;
        const parsed = Number(trimmed);
        if (Number.isFinite(parsed)) set(parsed);
    };

    return (
        <div className="floating-sidebar">
            <div className="coord-adjust-section">
                <h3>Coordinates</h3>
                <div className="coord-input-group">
                    <label>Latitude</label>
                    <input
                        type="number"
                        inputMode="decimal"
                        step="any"
                        min={-90}
                        max={90}
                        value={latitudeInput}
                        onChange={(e) => {
                            const raw = e.target.value;
                            setLatitudeInput(raw);
                            commitCoordinateIfValid(raw, setLatitude);
                        }}
                    />
                </div>
                <div className="coord-input-group">
                    <label>Longitude</label>
                    <input
                        type="number"
                        inputMode="decimal"
                        step="any"
                        min={-180}
                        max={180}
                        value={longitudeInput}
                        onChange={(e) => {
                            const raw = e.target.value;
                            setLongitudeInput(raw);
                            commitCoordinateIfValid(raw, setLongitude);
                        }}
                    />
                </div>
            </div>
        </div>
    );
};

export default FloatingSidebar;