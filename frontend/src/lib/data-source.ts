import { getDemoSnapshot } from "./demo-data";
import type { DashboardSnapshot } from "./types";

export interface DashboardDataSource {
  getSnapshot(leadHours: number): Promise<DashboardSnapshot>;
}

export const demoDataSource: DashboardDataSource = {
  async getSnapshot(leadHours) {
    return getDemoSnapshot(leadHours);
  },
};
