"""flat_power kit self-check: deck parse + champion load + map-peak path.
Usage: python val_fp.py <deck.inp> <model_dir>"""
import sys
from lpopt.config import load_config
deck, mdir = sys.argv[1], sys.argv[2]
c = load_config(deck)
print("DECK obj=%s workers=%s use_all_cores=%s host_reserve=%s harvest_maps=%s seed=%s"
      % (c.acquisition.objective, c.master.workers, c.master.use_all_cores,
         c.master.host_reserve, c.verify.harvest_maps, c.flow.random_seed))
from lpopt.model.model_api import PosValCnnBackend
b = PosValCnnBackend.from_dir(mdir, store_dir=c.model.store_dir,
                              library_id=c.model.library_id, device="cpu")
print("CHAMPION loaded=True has_predict_map_peak=%s" % hasattr(b, "predict_map_peak"))
