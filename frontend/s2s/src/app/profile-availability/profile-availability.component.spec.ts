import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProfileAvailabilityComponent } from './profile-availability.component';

describe('ProfileAvailabilityComponent', () => {
  let component: ProfileAvailabilityComponent;
  let fixture: ComponentFixture<ProfileAvailabilityComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ProfileAvailabilityComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ProfileAvailabilityComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
