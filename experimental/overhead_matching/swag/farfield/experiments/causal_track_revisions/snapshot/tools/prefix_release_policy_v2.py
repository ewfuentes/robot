"""Honor the tracker's explicit rejected-birth decision before releasing."""
from prefix_release_policy import OneShotPrefixRelease


def birth_usable(track):
    records=track['records']
    if not records or records[0]['action']!='birth':
        raise ValueError('Expected a recorded birth decision')
    health=records[0].get('health',{})
    if type(health.get('ok')) is not bool:
        raise ValueError('Birth health must be explicitly known')
    if not health['ok'] and not track['close_reason'].startswith('birth_'):
        raise ValueError('Rejected birth has inconsistent closure reason')
    return health['ok']


class ValidBirthPrefixRelease(OneShotPrefixRelease):
    def consider(self,*args,**kwargs):
        emission=super().consider(*args,**kwargs)
        if emission is None or not birth_usable(emission['track']):
            return None
        return emission
